# Peaks-over-threshold diagnostics: declustering, mean residual life, and
# parameter stability. Pure computation; plotting lives in the Makie extension.

using Dates: DateTime

"""
    decluster(times, levels, threshold; min_gap_hours=72) -> (peak_idx, excesses)

Group consecutive exceedances of `threshold` into clusters separated by at least
`min_gap_hours` below the threshold, and return the index and excess (level minus
threshold) of each cluster maximum.

Works on hourly (or sub-hourly) time series where `times` is a `Vector{DateTime}`.
For daily data without timestamps, use [`decluster_daily`](@ref).
"""
function decluster(
    times::AbstractVector{DateTime},
    levels::AbstractVector{<:Real},
    threshold::Real;
    min_gap_hours::Real=72,
)
    exceed_idx = findall(>(threshold), levels)
    isempty(exceed_idx) && return (Int[], Float64[])
    clusters = Vector{Vector{Int}}()
    current = [exceed_idx[1]]
    for i in 2:length(exceed_idx)
        gap_ms = (times[exceed_idx[i]] - times[exceed_idx[i-1]]).value
        if gap_ms / 3_600_000 > min_gap_hours
            push!(clusters, current)
            current = [exceed_idx[i]]
        else
            push!(current, exceed_idx[i])
        end
    end
    push!(clusters, current)
    peak_idx = [c[argmax(levels[c])] for c in clusters]
    return peak_idx, Float64[levels[i] - threshold for i in peak_idx]
end

"""
    decluster_daily(values, threshold; min_gap_days=3) -> (peak_idx, excesses)

Decluster a daily series by index gap rather than by timestamp.
"""
function decluster_daily(
    values::AbstractVector{<:Real}, threshold::Real; min_gap_days::Integer=3
)
    exceed_idx = findall(>(threshold), values)
    isempty(exceed_idx) && return (Int[], Float64[])
    clusters = Vector{Vector{Int}}()
    current = [exceed_idx[1]]
    for i in 2:length(exceed_idx)
        if exceed_idx[i] - exceed_idx[i-1] > min_gap_days
            push!(clusters, current)
            current = [exceed_idx[i]]
        else
            push!(current, exceed_idx[i])
        end
    end
    push!(clusters, current)
    peak_idx = [c[argmax(values[c])] for c in clusters]
    return peak_idx, Float64[values[i] - threshold for i in peak_idx]
end

"""
    mrl_data(values; n_thresholds=80, lo_quantile=0.5, hi_quantile=0.995)

Compute the mean residual life at a grid of thresholds.

Returns `(thresholds, means, lower, upper)` where `lower` and `upper` are
pointwise 95% confidence bounds. Thresholds with fewer than 5 exceedances
produce `NaN`.
"""
function mrl_data(
    values::AbstractVector{<:Real};
    n_thresholds::Integer=80,
    lo_quantile::Real=0.5,
    hi_quantile::Real=0.995,
)
    sorted = sort(collect(Float64, values))
    lo, hi = quantile(sorted, lo_quantile), quantile(sorted, hi_quantile)
    thresholds = range(lo, hi; length=n_thresholds)
    means = Vector{Float64}(undef, n_thresholds)
    lower = Vector{Float64}(undef, n_thresholds)
    upper = Vector{Float64}(undef, n_thresholds)
    for (k, u) in enumerate(thresholds)
        exc = sorted[sorted .> u] .- u
        n = length(exc)
        if n < 5
            means[k] = lower[k] = upper[k] = NaN
            continue
        end
        m = mean(exc)
        se = std(exc) / sqrt(n)
        means[k] = m
        lower[k] = m - 1.96 * se
        upper[k] = m + 1.96 * se
    end
    return collect(thresholds), means, lower, upper
end

"""
    stability_data(values; n_thresholds=40, lo_quantile=0.5, hi_quantile=0.98)

Fit the GPD at a grid of thresholds and return the reparameterized scale
\$\\sigma^* = \\sigma_u - \\xi u\$ and the shape \$\\xi\$ with approximate standard errors.

Returns `(thresholds, sigma_star, xi, sigma_star_se, xi_se)`. Thresholds where
the fit fails or has fewer than 10 exceedances produce `NaN`.

Uses [`gp_logpdf`](@ref) with Nelder-Mead optimization (no Turing overhead).
"""
function stability_data(
    values::AbstractVector{<:Real};
    n_thresholds::Integer=40,
    lo_quantile::Real=0.5,
    hi_quantile::Real=0.98,
)
    sorted = sort(collect(Float64, values))
    lo, hi = quantile(sorted, lo_quantile), quantile(sorted, hi_quantile)
    thresholds = range(lo, hi; length=n_thresholds)
    sigma_star = Vector{Float64}(undef, n_thresholds)
    xis = Vector{Float64}(undef, n_thresholds)
    sigma_star_se = Vector{Float64}(undef, n_thresholds)
    xi_se = Vector{Float64}(undef, n_thresholds)
    for (k, u) in enumerate(thresholds)
        exc = sorted[sorted .> u] .- u
        if length(exc) < 10
            sigma_star[k] = xis[k] = sigma_star_se[k] = xi_se[k] = NaN
            continue
        end
        try
            sigma, xi = _gpd_mle_quick(exc)
            n = length(exc)
            sigma_star[k] = sigma - xi * u
            xis[k] = xi
            sigma_star_se[k] = sigma / sqrt(n)
            xi_se[k] = (1 + xi) / sqrt(n)
        catch
            sigma_star[k] = xis[k] = sigma_star_se[k] = xi_se[k] = NaN
        end
    end
    return collect(thresholds), sigma_star, xis, sigma_star_se, xi_se
end

# Lightweight GPD MLE without Turing, for the stability plot where we fit
# dozens of times and don't need posteriors.
function _gpd_mle_quick(excesses::AbstractVector{<:Real})
    function neg_loglik(θ)
        log_σ, ξ = θ
        σ = exp(log_σ)
        return -sum(gp_logpdf(y, σ, ξ) for y in excesses)
    end
    init = [log(mean(excesses)), 0.1]
    result = _nelder_mead(neg_loglik, init)
    return exp(result[1]), result[2]
end

# Minimal Nelder-Mead so we don't need Optim as a dependency.
# Two parameters only, which is all the GPD needs.
function _nelder_mead(f, x0; maxiter=500, tol=1e-8)
    n = length(x0)
    simplex = [copy(x0) for _ in 1:n+1]
    for i in 2:n+1
        simplex[i][i-1] += 0.5
    end
    fvals = [f(s) for s in simplex]
    for _ in 1:maxiter
        order = sortperm(fvals)
        simplex .= simplex[order]
        fvals .= fvals[order]
        fvals[end] - fvals[1] < tol && break
        centroid = sum(simplex[1:n]) / n
        xr = centroid + (centroid - simplex[end])
        fr = f(xr)
        if fr < fvals[1]
            xe = centroid + 2 * (xr - centroid)
            fe = f(xe)
            if fe < fr
                simplex[end], fvals[end] = xe, fe
            else
                simplex[end], fvals[end] = xr, fr
            end
        elseif fr < fvals[n]
            simplex[end], fvals[end] = xr, fr
        else
            xc = centroid + 0.5 * (simplex[end] - centroid)
            fc = f(xc)
            if fc < fvals[end]
                simplex[end], fvals[end] = xc, fc
            else
                for i in 2:n+1
                    simplex[i] = simplex[1] + 0.5 * (simplex[i] - simplex[1])
                    fvals[i] = f(simplex[i])
                end
            end
        end
    end
    return simplex[argmin(fvals)]
end

"""
    pot_return_level(T, threshold, sigma, xi, lambda)

The `T`-year return level from a peaks-over-threshold fit with GPD scale `sigma`,
shape `xi`, threshold `threshold`, and Poisson exceedance rate `lambda` per year.
"""
function pot_return_level(T::Real, threshold::Real, sigma::Real, xi::Real, lambda::Real)
    if abs(xi) < 1e-7
        return threshold + sigma * log(T * lambda)
    else
        return threshold + (sigma / xi) * ((T * lambda)^xi - 1)
    end
end
