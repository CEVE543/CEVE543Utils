# L-moment estimation for the GEV, following Hosking, Wallis & Wood (1985),
# "Estimation of the generalized extreme-value distribution by the method of
# probability-weighted moments," Technometrics 27:251-261.
#
# The PWM algorithm is adapted from Extremes.jl (MIT, Jalbert, Farmer & Roy
# 2020) with thanks.

using SpecialFunctions: gamma

"""
    pwm(y, p, r, s) -> Float64

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

Algorithm: Hosking et al. (1985), polynomial approximation for the shape.
"""
function gevfit_lmom(y::AbstractVector{<:Real})
    b0 = pwm(y, 1, 0, 0)
    b1 = pwm(y, 1, 1, 0)
    b2 = pwm(y, 1, 2, 0)

    c = (2b1 - b0) / (3b2 - b0) - log(2) / log(3)
    k = 7.8590c + 2.9554c^2
    σ = k * (2b1 - b0) / ((1 - 2^(-k)) * gamma(1 + k))
    μ = b0 - σ / k * (1 - gamma(1 + k))
    ξ = -k

    return GeneralizedExtremeValue(μ, σ, ξ)
end
