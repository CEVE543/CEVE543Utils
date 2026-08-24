# The fitted object and the four entry points, named to match Extremes.jl.

"""
A fitted extreme value model.

`family` is `:gev` or `:gp`, `Xs` holds the design matrix used for each
parameter, and `estimate` is what Turing returned: a `ModeResult` from
[`gevfit`](@ref) and [`gpfit`](@ref), a chain from [`gevfitbayes`](@ref) and
[`gpfitbayes`](@ref).
"""
struct EVAFit{E}
    family::Symbol
    y::Vector{Float64}
    Xs::NamedTuple
    threshold::Float64
    estimate::E
end

"""
    params(fit) -> NamedTuple

The fitted coefficients for each parameter, intercept first.

Defined for the point estimates. A fit from [`gevfitbayes`](@ref) or
[`gpfitbayes`](@ref) holds a whole posterior rather than one value per
coefficient, so reach for `fit.estimate`, which is the chain.
"""
function params(fit::EVAFit)
    fit.estimate isa Turing.Optimisation.ModeResult || throw(
        ArgumentError(
            "this fit holds a posterior sample rather than a point estimate; " *
            "use `fit.estimate` to get the chain",
        ),
    )
    return NamedTuple(fit.estimate.params)
end

_stationary(fit::EVAFit) = all(X -> size(X, 2) == 1, values(fit.Xs))

"""
    getdistribution(fit) -> Vector{<:Distribution}

One distribution per observation, in the order the data came in.

A stationary fit gives every observation the same distribution, so the vector
has length one and `only(getdistribution(fit))` is the distribution. That is the
shape Extremes.jl returns, and the reason it is a vector at all is that a fit
with covariates has a different distribution at every row.
"""
function getdistribution(fit::EVAFit)
    β = params(fit)
    rows = _stationary(fit) ? (1:1) : eachindex(fit.y)
    if fit.family === :gev
        return [
            GeneralizedExtremeValue(
                row_value(fit.Xs.location, i, β.β_location),
                exp(row_value(fit.Xs.logscale, i, β.β_logscale)),
                row_value(fit.Xs.shape, i, β.β_shape),
            ) for i in rows
        ]
    end
    return [
        GeneralizedPareto(
            fit.threshold,
            exp(row_value(fit.Xs.logscale, i, β.β_logscale)),
            row_value(fit.Xs.shape, i, β.β_shape),
        ) for i in rows
    ]
end

"""
    returnlevel(fit, T) -> Vector

The level exceeded with probability `1 / T` in one period, one entry per
distribution [`getdistribution`](@ref) returns.

A peaks over threshold fit carries the threshold in its distributions, so this
is already a level on the original scale.
"""
returnlevel(fit::EVAFit, T) = [quantile(d, 1 - 1 / T) for d in getdistribution(fit)]

"""
    loglike(fit) -> Real

The log density at the fitted parameters.
"""
loglike(fit::EVAFit) = fit.estimate.lp

# Every slope starts at zero and every intercept at a moment estimate, so the
# optimizer opens on the stationary fit whatever the covariates are.
#
# The shape is the exception: it starts a little away from zero. Both densities
# fall back to their ξ = 0 limit near zero, and neither limit contains ξ, so the
# gradient with respect to the shape is identically zero there. Starting exactly
# at zero means the optimizer reports success without ever having moved it.
const SHAPE_INIT = 0.1

_zeros(X) = zeros(size(X, 2))

function _shape_start(X)
    β = zeros(size(X, 2))
    β[1] = SHAPE_INIT
    return β
end

function _gev_init(y, Xs)
    location, logscale = _zeros(Xs.location), _zeros(Xs.logscale)
    location[1], logscale[1] = median(y), log(std(y))
    return DynamicPPL.InitFromParams((
        β_location=location, β_logscale=logscale, β_shape=_shape_start(Xs.shape)
    ))
end

function _gp_init(excess, Xs)
    logscale = _zeros(Xs.logscale)
    logscale[1] = log(mean(excess))
    return DynamicPPL.InitFromParams((β_logscale=logscale, β_shape=_shape_start(Xs.shape)))
end

function _gev_matrices(n, loc, logscale, shape)
    return (
        location=design_matrix(n, loc),
        logscale=design_matrix(n, logscale),
        shape=design_matrix(n, shape),
    )
end

function _exceedances(y, threshold, logscalecov, shapecov)
    keep = findall(>(threshold), y)
    isempty(keep) && throw(ArgumentError("no values above threshold $threshold"))
    excess = y[keep] .- threshold
    sub(c) = isnothing(c) ? nothing : [collect(v)[keep] for v in c]
    Xs = (
        logscale=design_matrix(length(excess), sub(logscalecov)),
        shape=design_matrix(length(excess), sub(shapecov)),
    )
    return excess, Xs
end

"""
    gevfit(y; locationcov, logscalecov, shapecov, prior_scale, kwargs...)

Fit a generalized extreme value distribution to block maxima by maximum
likelihood, through Turing's mode estimation.

Pass covariates to make the fit nonstationary. Each parameter gets its own
intercept, so `locationcov=[years]` moves the location with the year and leaves
the scale and shape fixed.

Use [`getdistribution`](@ref) to turn the result into distributions and
[`returnlevel`](@ref) to read a level off it. For the posterior rather than a
point estimate, see [`gevfitbayes`](@ref).
"""
function gevfit(
    y::AbstractVector;
    locationcov=nothing,
    logscalecov=nothing,
    shapecov=nothing,
    prior_scale=100.0,
    kwargs...,
)
    y = collect(Float64, y)
    Xs = _gev_matrices(length(y), locationcov, logscalecov, shapecov)
    model = gev_model(y, Xs.location, Xs.logscale, Xs.shape, prior_scale)
    estimate = maximum_likelihood(model; initial_params=_gev_init(y, Xs), kwargs...)
    return EVAFit(:gev, y, Xs, NaN, estimate)
end

"""
    gpfit(y, threshold; logscalecov, shapecov, prior_scale, kwargs...)

Fit a generalized Pareto distribution to the amounts by which `y` exceeds
`threshold`, by maximum likelihood.

Values at or below the threshold are dropped, and covariates are subset to the
exceedances alongside the data. [`getdistribution`](@ref) returns distributions
carrying `threshold` as their location, so a quantile read off one is a level on
the original scale.
"""
function gpfit(
    y::AbstractVector,
    threshold::Real;
    logscalecov=nothing,
    shapecov=nothing,
    prior_scale=100.0,
    kwargs...,
)
    y = collect(Float64, y)
    excess, Xs = _exceedances(y, threshold, logscalecov, shapecov)
    model = gp_model(excess, Xs.logscale, Xs.shape, prior_scale)
    estimate = maximum_likelihood(model; initial_params=_gp_init(excess, Xs), kwargs...)
    return EVAFit(:gp, excess, Xs, Float64(threshold), estimate)
end

"""
    gevfitbayes(y; locationcov, logscalecov, shapecov, n_samples, n_chains, rng)

Sample the posterior with NUTS instead of taking a point estimate.

The model, the covariate keywords, and the priors are the ones [`gevfit`](@ref)
uses, so a nonstationary fit reads the same either way. `fit.estimate` is the
chain.
"""
function gevfitbayes(
    y::AbstractVector;
    locationcov=nothing,
    logscalecov=nothing,
    shapecov=nothing,
    prior_scale=100.0,
    n_samples=1_000,
    n_chains=4,
    rng=Random.default_rng(),
    kwargs...,
)
    y = collect(Float64, y)
    Xs = _gev_matrices(length(y), locationcov, logscalecov, shapecov)
    model = gev_model(y, Xs.location, Xs.logscale, Xs.shape, prior_scale)
    chain = sample(rng, model, NUTS(), MCMCThreads(), n_samples, n_chains; kwargs...)
    return EVAFit(:gev, y, Xs, NaN, chain)
end

"""
    gpfitbayes(y, threshold; logscalecov, shapecov, n_samples, n_chains, rng)

Sample the posterior of a peaks over threshold fit with NUTS.
"""
function gpfitbayes(
    y::AbstractVector,
    threshold::Real;
    logscalecov=nothing,
    shapecov=nothing,
    prior_scale=100.0,
    n_samples=1_000,
    n_chains=4,
    rng=Random.default_rng(),
    kwargs...,
)
    y = collect(Float64, y)
    excess, Xs = _exceedances(y, threshold, logscalecov, shapecov)
    model = gp_model(excess, Xs.logscale, Xs.shape, prior_scale)
    chain = sample(rng, model, NUTS(), MCMCThreads(), n_samples, n_chains; kwargs...)
    return EVAFit(:gp, excess, Xs, Float64(threshold), chain)
end
