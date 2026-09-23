module CEVE543UtilsMakieExt

using CEVE543Utils: CEVE543Utils, mrl_data, stability_data
using Makie

const TICKS = [1, 2, 5, 10, 25, 100, 1000]

"""
    return_period_axis!(ax; ticks) -> Axis

Put a log10 return period on the horizontal axis of `ax`, with labelled ticks at
the return periods these plots usually mark.

Everything else about `ax` is left alone, so set the vertical label and the
title when you build it.

Prefer calling this after plotting into `ax`. A log scale cannot be applied
while an axis still holds the default limits it has with nothing in it, since
those start at zero, so styling an empty axis pins the horizontal limits to the
span of `ticks` and gives up autoscaling.
"""
function CEVE543Utils.return_period_axis!(ax; ticks=TICKS)
    ax.xlabel = L"\text{return period } T \text{ (years)}"
    ax.xticks = (ticks, string.(ticks))
    isempty(ax.scene.plots) && xlims!(ax, extrema(ticks)...)
    ax.xscale = log10
    return ax
end

"""
    return_period_axis(position; ticks, kwargs...) -> Axis

Build an `Axis` in `position` and style it with `return_period_axis!`.

`position` is a figure slot, such as `fig[1, 1]`. Anything else you pass goes to
`Axis`, so `ylabel` and `title` work as usual.
"""
function CEVE543Utils.return_period_axis(position; ticks=TICKS, kwargs...)
    return CEVE543Utils.return_period_axis!(Axis(position; kwargs...); ticks=ticks)
end

"""
    mrl_plot(values; kwargs...) -> Figure

Mean residual life plot with 95% confidence band.
Keyword arguments are passed through to [`mrl_data`](@ref).
"""
function CEVE543Utils.mrl_plot(values; color=Makie.wong_colors()[1], figsize=(800, 340), kwargs...)
    t, m, lo_band, hi_band = mrl_data(values; kwargs...)
    fig = Figure(; size=figsize)
    ax = Axis(fig[1, 1]; xlabel="Threshold", ylabel="Mean excess")
    band!(ax, t, lo_band, hi_band; color=(color, 0.2))
    lines!(ax, t, m; color=color, linewidth=2)
    return fig
end

"""
    stability_plot(values; kwargs...) -> Figure

Parameter stability plot: reparameterized scale σ* and shape ξ against threshold,
with 95% confidence bands. Keyword arguments are passed through to
[`stability_data`](@ref).
"""
function CEVE543Utils.stability_plot(values; figsize=(800, 500), kwargs...)
    colors = Makie.wong_colors()
    t, ss, xi, ss_se, xi_se = stability_data(values; kwargs...)
    fig = Figure(; size=figsize)
    ax1 = Axis(fig[1, 1]; ylabel=L"\sigma^* = \sigma_u - \xi u")
    band!(ax1, t, ss .- 1.96 .* ss_se, ss .+ 1.96 .* ss_se; color=(colors[1], 0.2))
    scatter!(ax1, t, ss; color=colors[1], markersize=5)
    ax2 = Axis(fig[2, 1]; xlabel="Threshold", ylabel=L"\xi")
    band!(ax2, t, xi .- 1.96 .* xi_se, xi .+ 1.96 .* xi_se; color=(colors[2], 0.2))
    scatter!(ax2, t, xi; color=colors[2], markersize=5)
    return fig
end

end # module
