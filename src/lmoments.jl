# L-moment estimation for the GEV, following Hosking, Wallis & Wood (1985),
# "Estimation of the generalized extreme-value distribution by the method of
# probability-weighted moments," Technometrics 27:251-261.
#
# The PWM algorithm is adapted from Extremes.jl (MIT, Jalbert, Farmer & Roy
# 2020) with thanks.

using SpecialFunctions: gamma
using Base.MathConstants: eulergamma

"""
    pwm(y, p, r, s)

Empirical probability weighted moment ``M_{p,r,s}`` using the unbiased
estimator of Landwehr, Matalas & Wallis (1979).
"""
function pwm(y::AbstractVector{<:Real}, p::Int, r::Int, s::Int)
    x = sort(y)
    n = length(x)
    return sum(
        x[i]^p * binomial(i - 1, r) / binomial(n - 1, r) *
        binomial(n - i, s) / binomial(n - 1, s)
        for i in 1:n
    ) / n
end

"""
    gevfit_lmom(y) -> GeneralizedExtremeValue

Fit a GEV to `y` by the method of probability weighted moments (L-moments).

Returns a `Distributions.GeneralizedExtremeValue`. This is a stationary,
closed-form estimator: no iteration, no priors, no optimizer. It gives a
different answer from MLE, and comparing the two is one reason to show it.

Algorithm: Hosking et al. (1985); the polynomial for the shape holds for
`-0.5 <= ξ <= 0.5`. Requires `length(y) >= 3`.
"""
function gevfit_lmom(y::AbstractVector{<:Real})
    length(y) >= 3 || throw(
        ArgumentError("gevfit_lmom needs at least 3 observations, got $(length(y))")
    )
    return gev_from_pwm(pwm(y, 1, 0, 0), pwm(y, 1, 1, 0), pwm(y, 1, 2, 0))
end

"""
    gev_from_pwm(b0, b1, b2) -> GeneralizedExtremeValue

The GEV whose first three probability weighted moments are `b0, b1, b2`
(Hosking et al. 1985, eqs. 14-16).
"""
function gev_from_pwm(b0::Real, b1::Real, b2::Real)
    c = (2b1 - b0) / (3b2 - b0) - log(2) / log(3)
    k = 7.8590c + 2.9554c^2
    if abs(k) < 1e-8
        # Gumbel limit: σ = λ₂ / log 2, μ = λ₁ - γσ
        σ = (2b1 - b0) / log(2)
        return GeneralizedExtremeValue(b0 - eulergamma * σ, σ, 0.0)
    end
    σ = k * (2b1 - b0) / ((1 - 2^(-k)) * gamma(1 + k))
    μ = b0 - σ / k * (1 - gamma(1 + k))
    return GeneralizedExtremeValue(μ, σ, -k)
end
