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

@model function gev_model(y, X_location, X_logscale, X_shape, prior_scale)
    β_location ~ MvNormal(zeros(size(X_location, 2)), prior_scale * I)
    β_logscale ~ MvNormal(zeros(size(X_logscale, 2)), prior_scale * I)
    β_shape ~ MvNormal(zeros(size(X_shape, 2)), 0.25 * I)

    @inbounds for i in eachindex(y)
        μ = row_value(X_location, i, β_location)
        σ = exp(row_value(X_logscale, i, β_logscale))
        ξ = row_value(X_shape, i, β_shape)
        y[i] ~ GEVKernel(μ, σ, ξ)
    end
end

@model function gp_model(excess, X_logscale, X_shape, prior_scale)
    β_logscale ~ MvNormal(zeros(size(X_logscale, 2)), prior_scale * I)
    β_shape ~ MvNormal(zeros(size(X_shape, 2)), 0.25 * I)

    @inbounds for i in eachindex(excess)
        σ = exp(row_value(X_logscale, i, β_logscale))
        ξ = row_value(X_shape, i, β_shape)
        excess[i] ~ GPKernel(σ, ξ)
    end
end
