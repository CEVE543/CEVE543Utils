# Log densities for the two extreme value families.
#
# Written out rather than taken from Distributions.jl so a parameter set outside
# the support returns `-Inf` instead of throwing. A sampler walks into that
# region routinely, and an exception there costs far more than a branch.
#
# Both densities are written around one quantity,
#
#     w = log1p(ξz) / ξ,
#
# because that is where the shape divides into a logarithm and so is the only
# place the ξ → 0 limit is delicate. Writing them this way means one series
# expansion covers every appearance of the shape.

"""
    log1p_over(z, ξ)

`log1p(ξz) / ξ`, which tends to `z` as `ξ` tends to zero.

Near zero the quotient divides a vanishing logarithm by a vanishing shape, so
it is evaluated as the series

``\\log(1 + u)/ξ = z(1 - u/2 + u^2/3 - u^3/4 + \\cdots)``, ``u = ξz``,

which is smooth through zero and carries the shape's derivative with it. A
branch onto the `ξ = 0` limit instead would drop `ξ` from the expression, and
the derivative with respect to a parameter that no longer appears is zero, which
an optimizer reads as convergence in that coordinate.

The series is taken for `|u| <= 0.01`, where its truncation error is already at
machine precision, rather than at the much smaller threshold that the direct
form's own cancellation would suggest.
"""
@inline function log1p_over(z, ξ)
    u = ξ * z
    if abs(u) <= 0.01
        return z * (1 - u / 2 + u^2 / 3 - u^3 / 4 + u^4 / 5 - u^5 / 6)
    end
    return log1p(u) / ξ
end

"""
    gev_logpdf(y, μ, σ, ξ)

Log density of the generalized extreme value distribution, returning `-Inf`
rather than throwing when `y` is outside the support or `σ` is not positive.

The Gumbel is the `ξ = 0` case and needs no branch of its own: the density is
written through [`log1p_over`](@ref), whose limit supplies it.
"""
@inline function gev_logpdf(y, μ, σ, ξ)
    σ <= 0 && return oftype(σ / one(y), -Inf)
    z = (y - μ) / σ
    u = ξ * z
    # 1 + ξz is the support condition, and `log1p` would raise rather than
    # return a non-finite value once it fails, so the test comes first.
    u <= -1 && return oftype(u, -Inf)
    w = log1p_over(z, ξ)
    return -log(σ) - log1p(u) - w - exp(-w)
end

"""
    gp_logpdf(y, σ, ξ)

Log density of the generalized Pareto distribution over an exceedance `y`,
measured from the threshold rather than from zero on the original scale.

The exponential is the `ξ = 0` case and, as in [`gev_logpdf`](@ref), arrives as
a limit rather than a branch.
"""
@inline function gp_logpdf(y, σ, ξ)
    σ <= 0 && return oftype(σ / one(y), -Inf)
    z = y / σ
    z < 0 && return oftype(z, -Inf)   # support is y >= 0
    u = ξ * z
    u <= -1 && return oftype(u, -Inf)
    return -log(σ) - log1p(u) - log1p_over(z, ξ)
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
