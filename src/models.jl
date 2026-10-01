# Design matrices and the two Turing models.
#
# The variables are named `β_location`, `β_logscale` and `β_shape` so that
# `NamedTuple(result.params)` comes back keyed the way `params` reports it, with
# no index arithmetic in between.

"""
    design_matrix(n, covariates) -> Matrix

A column of ones followed by one column per covariate, so the first coefficient
is always the intercept and a fit with no covariates is the stationary one.

`covariates` is `nothing`, a vector of vectors, or a matrix whose columns are
the covariates.
"""
function design_matrix(n::Integer, covariates)
    isnothing(covariates) && return ones(n, 1)
    cols = covariates isa AbstractMatrix ? collect(eachcol(covariates)) : covariates
    isempty(cols) && return ones(n, 1)
    all(c -> length(c) == n, cols) ||
        throw(DimensionMismatch("every covariate must have $n entries, as the data does"))
    return hcat(ones(n), reduce(hcat, [collect(Float64, c) for c in cols]))
end

"""
    standardize(X) -> (X, center, scale)

Replace each covariate column `x` of a design matrix by `(x - mean(x)) / std(x)`,
leaving the intercept column alone. A column with `std(x) <= 1e-8 max(|mean(x)|, 1)`
(constant up to rounding, or a single row) gets `scale = 1`.
[`params`](@ref) uses `center` and `scale` to undo the transform.
"""
function standardize(X::AbstractMatrix)
    center = vec(mean(X[:, 2:end]; dims=1))
    scale = vec(std(X[:, 2:end]; dims=1))
    constant = .!(scale .> 1e-8 .* max.(abs.(center), 1))   # also catches NaN
    scale[constant] .= 1
    Z = copy(X)
    Z[:, 2:end] .= (X[:, 2:end] .- center') ./ scale'
    return (X=Z, center=center, scale=scale)
end

# One row of `X * β`, written as a scalar loop. `X * β` inside an automatic
# differentiation pass allocates a fresh dual array on every gradient
# evaluation, which is what makes the obvious nonstationary model crawl.
@inline function row_value(X, i, β)
    acc = zero(eltype(β))
    @inbounds for j in eachindex(β)
        acc += X[i, j] * β[j]
    end
    return acc
end

"""
    coefficient_prior(k, intercept_prior, slope_prior)

Prior on a coefficient vector of length `k`: `intercept_prior` on the intercept,
`slope_prior` on each of the `k - 1` slopes of standardized covariates.
"""
coefficient_prior(k, intercept_prior, slope_prior) =
    arraydist([intercept_prior; fill(slope_prior, k - 1)])

# A quantile prior `q` holds the design rows of one observation, a return period
# `q.period` and a distribution `q.belief`; with z_q the q.period-year level at
# those rows, it adds log q.belief(z_q) to the log prior, so mode estimation
# leaves it out.

@model function gev_model(
    y, X_location, X_logscale, X_shape, trend_priors, shape_prior, quantile_priors
)
    β_location ~ coefficient_prior(size(X_location, 2), Flat(), trend_priors.location)
    β_logscale ~ coefficient_prior(size(X_logscale, 2), Flat(), trend_priors.logscale)
    β_shape ~ coefficient_prior(size(X_shape, 2), shape_prior, trend_priors.shape)

    for q in quantile_priors
        d = GeneralizedExtremeValue(
            dot(q.location, β_location), exp(dot(q.logscale, β_logscale)), dot(q.shape, β_shape)
        )
        Turing.@addlogprob! (; logprior=logpdf(q.belief, returnlevel(d, q.period)))
    end

    @inbounds for i in eachindex(y)
        μ = row_value(X_location, i, β_location)
        σ = exp(row_value(X_logscale, i, β_logscale))
        ξ = row_value(X_shape, i, β_shape)
        y[i] ~ GEVKernel(μ, σ, ξ)
    end
end

@model function gp_model(
    excess, X_logscale, X_shape, trend_priors, shape_prior, quantile_priors, threshold, rate
)
    β_logscale ~ coefficient_prior(size(X_logscale, 2), Flat(), trend_priors.logscale)
    β_shape ~ coefficient_prior(size(X_shape, 2), shape_prior, trend_priors.shape)

    for q in quantile_priors
        d = GeneralizedPareto(threshold, exp(dot(q.logscale, β_logscale)), dot(q.shape, β_shape))
        Turing.@addlogprob! (;
            logprior=logpdf(q.belief, returnlevel(d, q.period; rate=rate))
        )
    end

    @inbounds for i in eachindex(excess)
        σ = exp(row_value(X_logscale, i, β_logscale))
        ξ = row_value(X_shape, i, β_shape)
        excess[i] ~ GPKernel(σ, ξ)
    end
end
