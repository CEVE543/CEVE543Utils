# The fitted object and the four entry points, named to match Extremes.jl.

"""
A fitted extreme value model.

`family` is `:gev` or `:gp`, `Xs` holds, for each parameter, the standardized
design matrix as `X` with its `center` and `scale`, and `estimate` is what Turing returned: a `ModeResult` from
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

The fitted coefficients for each parameter, intercept first, on the scale of the
covariates as passed in. The fit works on standardized covariates
`z = (x - c) / s`, so `fit.estimate` holds `β` with `β₀ + Σ βⱼ zⱼ`; this returns
the equivalent `(β₀ - Σ βⱼ cⱼ / sⱼ, β₁ / s₁, …)`.

Point estimates only; for a posterior use [`posterior_distributions`](@ref).
"""
function params(fit::EVAFit)
    raw = _raw_params(fit)
    return NamedTuple{keys(raw)}(
        Tuple(_original_scale(raw[k], fit.Xs[_design_key(k)]) for k in keys(raw))
    )
end

# The coefficients as the optimizer left them, on the standardized scale.
function _raw_params(fit::EVAFit)
    fit.estimate isa Turing.Optimisation.ModeResult || throw(
        ArgumentError(
            "this fit holds a posterior sample rather than a point estimate; " *
            "use `posterior_distributions`",
        ),
    )
    return NamedTuple(fit.estimate.params)
end

# `β_location` was fitted against `Xs.location`, and so on.
_design_key(name::Symbol) = Symbol(chopprefix(string(name), "β_"))

function _original_scale(β, design)
    slopes = β[2:end] ./ design.scale
    return vcat(β[1] - sum(slopes .* design.center), slopes)
end

_stationary(fit::EVAFit) = all(d -> size(d.X, 2) == 1, values(fit.Xs))

"""
    getdistribution(fit) -> Vector{<:Distribution}

One distribution per observation, in the order the data came in; for a peaks
over threshold fit, one per exceedance.

A stationary fit gives every observation the same distribution, so the vector
has length one and `only(getdistribution(fit))` is the distribution. That is the
shape Extremes.jl returns, and the reason it is a vector at all is that a fit
with covariates has a different distribution at every row.
"""
getdistribution(fit::EVAFit) = _distributions(fit, _raw_params(fit))

"""
    posterior_distributions(fit) -> Vector{Vector{<:Distribution}}

One entry per posterior draw of a fit from [`gevfitbayes`](@ref) or
[`gpfitbayes`](@ref), each shaped as [`getdistribution`](@ref) shapes a point
estimate: length one for a stationary fit, one distribution per observation for
a fit with covariates.

    draws = posterior_distributions(fit)
    [returnlevel(only(d), 100; rate=λ) for d in draws]   # 100-year level, per draw
"""
function posterior_distributions(fit::EVAFit)
    chain = fit.estimate
    chain isa Turing.Optimisation.ModeResult && throw(
        ArgumentError(
            "this fit holds a point estimate rather than a posterior sample; " *
            "use `getdistribution`",
        ),
    )
    # Turing returns a FlexiChain indexed by variable name, and each entry is the
    # whole coefficient vector for one draw.
    names = fit.family === :gev ? (:β_location, :β_logscale, :β_shape) : (:β_logscale, :β_shape)
    columns = [vec(chain[name]) for name in names]
    return [
        _distributions(fit, NamedTuple{names}(Tuple(c[k] for c in columns))) for
        k in eachindex(first(columns))
    ]
end

# The distributions implied by one set of coefficients, shared by the point
# estimate and by every posterior draw.
function _distributions(fit::EVAFit, β)
    rows = _stationary(fit) ? (1:1) : eachindex(fit.y)
    if fit.family === :gev
        return [
            GeneralizedExtremeValue(
                row_value(fit.Xs.location.X, i, β.β_location),
                exp(row_value(fit.Xs.logscale.X, i, β.β_logscale)),
                row_value(fit.Xs.shape.X, i, β.β_shape),
            ) for i in rows
        ]
    end
    return [
        GeneralizedPareto(
            fit.threshold,
            exp(row_value(fit.Xs.logscale.X, i, β.β_logscale)),
            row_value(fit.Xs.shape.X, i, β.β_shape),
        ) for i in rows
    ]
end

"""
    returnlevel(fit, T; rate) -> Vector

The `T`-year return level, one entry per distribution [`getdistribution`](@ref)
returns.

A peaks over threshold fit needs `rate`, the number of exceedances per year,
because its distributions describe one exceedance rather than one year. It
carries the threshold in its distributions, so the result is already a level on
the original scale.
"""
function returnlevel(fit::EVAFit, T; rate=nothing)
    if fit.family === :gp
        isnothing(rate) && throw(
            ArgumentError(
                "a peaks over threshold fit needs `rate`, the exceedances per year, " *
                "to turn a return period in years into a level",
            ),
        )
        return [returnlevel(d, T; rate=rate) for d in getdistribution(fit)]
    end
    return [returnlevel(d, T) for d in getdistribution(fit)]
end

"""
    returnlevel(d::GeneralizedExtremeValue, T)
    returnlevel(d::GeneralizedPareto, T; rate)

The `T`-year return level of one distribution, such as one entry of
[`getdistribution`](@ref) or of [`posterior_distributions`](@ref).

A GEV fitted to annual maxima describes one year, so the level is its
`1 - 1/T` quantile. A GPD describes one exceedance of its threshold, which is
its location, so it needs `rate`, the exceedances per year; the level is
[`pot_return_level`](@ref).
"""
returnlevel(d::GeneralizedExtremeValue, T) = quantile(d, 1 - 1 / T)
returnlevel(d::GeneralizedPareto, T; rate) = pot_return_level(T, d.μ, d.σ, d.ξ, rate)

"""
    loglike(fit) -> Real

The log likelihood at the point estimate. Point estimates only.
"""
function loglike(fit::EVAFit)
    fit.estimate isa Turing.Optimisation.ModeResult || throw(
        ArgumentError("loglike is defined for a point estimate; this fit holds a posterior sample")
    )
    return fit.estimate.lp
end

# Every slope starts at zero and every intercept at a moment estimate, so the
# optimizer opens on the stationary fit whatever the covariates are.
#
# The shape is the exception: it starts a little away from zero. The densities
# keep a derivative in ξ through zero (see `log1p_over`), but the likelihood is
# flat enough there that an optimizer started exactly at zero can stop early.
const SHAPE_INIT = 0.1

_zeros(design) = zeros(size(design.X, 2))

function _shape_start(design)
    β = zeros(size(design.X, 2))
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
        location=standardize(design_matrix(n, loc)),
        logscale=standardize(design_matrix(n, logscale)),
        shape=standardize(design_matrix(n, shape)),
    )
end

function _exceedances(y, threshold, logscalecov, shapecov)
    keep = findall(>(threshold), y)
    isempty(keep) && throw(ArgumentError("no values above threshold $threshold"))
    excess = y[keep] .- threshold
    sub(c) = isnothing(c) ? nothing : [collect(v)[keep] for v in _covariate_columns(c)]
    Xs = (
        logscale=standardize(design_matrix(length(excess), sub(logscalecov))),
        shape=standardize(design_matrix(length(excess), sub(shapecov))),
    )
    return excess, Xs
end

_covariate_columns(c::AbstractMatrix) = eachcol(c)
_covariate_columns(c) = c

"""
    gevfit(y; locationcov, logscalecov, shapecov, kwargs...)

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
    kwargs...,
)
    y = collect(Float64, y)
    Xs = _gev_matrices(length(y), locationcov, logscalecov, shapecov)
    # `maximum_likelihood` ignores the prior; 1.0 is a placeholder.
    model = gev_model(y, Xs.location.X, Xs.logscale.X, Xs.shape.X, 1.0)
    estimate = maximum_likelihood(model; initial_params=_gev_init(y, Xs), kwargs...)
    return EVAFit(:gev, y, Xs, NaN, estimate)
end

"""
    gpfit(y, threshold; logscalecov, shapecov, kwargs...)

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
    kwargs...,
)
    y = collect(Float64, y)
    excess, Xs = _exceedances(y, threshold, logscalecov, shapecov)
    # `maximum_likelihood` ignores the prior; 1.0 is a placeholder.
    model = gp_model(excess, Xs.logscale.X, Xs.shape.X, 1.0)
    estimate = maximum_likelihood(model; initial_params=_gp_init(excess, Xs), kwargs...)
    return EVAFit(:gp, excess, Xs, Float64(threshold), estimate)
end

"""
    gevfitbayes(y; locationcov, logscalecov, shapecov, prior_scale, n_samples, n_chains, rng, sampler)

Sample the posterior with NUTS instead of taking a point estimate.

The model and the covariate keywords are the ones [`gevfit`](@ref) uses, with
the priors below on top, so a nonstationary fit reads the same either way.
`fit.estimate` is the chain, and [`posterior_distributions`](@ref) turns it
into distributions.

Covariates are standardized (see [`standardize`](@ref)), and `fit.estimate` is
on that scale. Priors: `N(0, prior_scale)` on every location and log-scale
coefficient, so the default 100 is a standard deviation of 10 per standard
deviation of the covariate; `N(0, 0.25)` on the shape coefficients.

`sampler` defaults to `NUTS()`. The GEV support moves with its parameters, so
divergences are common at the default step size; `sampler=NUTS(0.99)` takes
smaller steps and usually removes them.
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
    sampler=NUTS(),
    kwargs...,
)
    y = collect(Float64, y)
    Xs = _gev_matrices(length(y), locationcov, logscalecov, shapecov)
    model = gev_model(y, Xs.location.X, Xs.logscale.X, Xs.shape.X, prior_scale)
    chain = sample(rng, model, sampler, MCMCThreads(), n_samples, n_chains; kwargs...)
    return EVAFit(:gev, y, Xs, NaN, chain)
end

"""
    gpfitbayes(y, threshold; logscalecov, shapecov, prior_scale, n_samples, n_chains, rng, sampler)

Sample the posterior of a peaks over threshold fit with NUTS.

Covariates are standardized and `fit.estimate` is on that scale, as in
[`gevfitbayes`](@ref). Priors: `N(0, prior_scale)` on every log-scale
coefficient, `N(0, 0.25)` on the shape coefficients. [`posterior_distributions`](@ref)
turns the chain into distributions. `sampler` is passed to Turing's `sample` and
defaults to `NUTS()`, as in [`gevfitbayes`](@ref).
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
    sampler=NUTS(),
    kwargs...,
)
    y = collect(Float64, y)
    excess, Xs = _exceedances(y, threshold, logscalecov, shapecov)
    model = gp_model(excess, Xs.logscale.X, Xs.shape.X, prior_scale)
    chain = sample(rng, model, sampler, MCMCThreads(), n_samples, n_chains; kwargs...)
    return EVAFit(:gp, excess, Xs, Float64(threshold), chain)
end
