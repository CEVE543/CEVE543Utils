module CEVE543UtilsMakieExt

using CEVE543Utils: CEVE543Utils
using Makie: Axis, xlims!

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
    ax.xlabel = "return period T (years)"
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

end # module
