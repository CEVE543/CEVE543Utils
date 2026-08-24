# Log densities for the two extreme value families.
#
# Written out rather than taken from Distributions.jl so a parameter set outside
# the support returns `-Inf` instead of throwing. A sampler walks into that
# region routinely, and an exception there costs far more than a branch.

"""
    gev_logpdf(y, μ, σ, ξ)

Log density of the generalized extreme value distribution, returning `-Inf`
rather than throwing when `y` is outside the support or `σ` is not positive.

The `ξ = 0` limit is the Gumbel density, taken whenever `|ξ|` is small enough
that the general form loses accuracy.
"""
@inline function gev_logpdf(y, μ, σ, ξ)
    σ <= 0 && return oftype(σ / one(y), -Inf)
    z = (y - μ) / σ
    abs(ξ) < 1e-7 && return -log(σ) - z - exp(-z)
    v = 1 + ξ * z
    v <= 0 && return oftype(v, -Inf)
    return -log(σ) - (1 + 1 / ξ) * log(v) - v^(-1 / ξ)
end

"""
    gp_logpdf(y, σ, ξ)

Log density of the generalized Pareto distribution over an exceedance `y`,
measured from the threshold rather than from zero on the original scale.

The `ξ = 0` limit is the exponential density.
"""
@inline function gp_logpdf(y, σ, ξ)
    σ <= 0 && return oftype(σ / one(y), -Inf)
    z = y / σ
    abs(ξ) < 1e-7 && return -log(σ) - z
    v = 1 + ξ * z
    v <= 0 && return oftype(v, -Inf)
    return -log(σ) - (1 + 1 / ξ) * log(v)
end

# Thin wrappers so the kernels can be reached with `~` inside a Turing model.
# Going through `~` rather than `@addlogprob!` is what keeps mode estimation
# honest: `maximum_likelihood` has to tell the likelihood from the prior, and a
# term added by hand lands in both.

struct GEVKernel{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ξ::T
end

struct GPKernel{T<:Real} <: ContinuousUnivariateDistribution
    σ::T
    ξ::T
end

Distributions.logpdf(d::GEVKernel, y::Real) = gev_logpdf(y, d.μ, d.σ, d.ξ)
Distributions.logpdf(d::GPKernel, y::Real) = gp_logpdf(y, d.σ, d.ξ)
