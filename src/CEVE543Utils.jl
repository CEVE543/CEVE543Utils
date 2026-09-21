"""
Extreme value fitting for CEVE 543, with the interface of Extremes.jl over a
Turing model.

`gevfit`, `gpfit`, `getdistribution` and `returnlevel` mean what they mean in
Extremes.jl, and covariates are named the same way. What differs is underneath:
each family is one Turing model, entered either by mode estimation for a point
estimate or by NUTS for the posterior, so covariates reach both.

    fit = gevfit(annmax)                          # stationary, maximum likelihood
    gev = only(getdistribution(fit))              # one distribution
    returnlevel(fit, 100)                         # the 100-year level

    fit = gevfit(annmax; locationcov=[years])     # location moves with the year
    getdistribution(fit)                          # one distribution per year

`load_water_level` downloads a NOAA tide gauge's record and `AnnMaxRecord` takes
its annual maxima, so a fit can be run against any station on the network.
Levels carry Unitful units, and `ustrip` hands plain numbers to a fit:

    record = load_water_level("8638610")             # Sewells Point, VA
    annmax = AnnMaxRecord(record; detrend=:msl)
    gevfit(ustrip.(u"ft", levels(annmax)))

Station ids are listed at <https://tidesandcurrents.noaa.gov/stations.html>.

Plot helpers live in an extension: `plotting_positions` is always available, and
`return_period_axis!` appears once Makie is loaded.
"""
module CEVE543Utils

using Distributions
using LinearAlgebra
using Random
using Statistics
using Tables
using Turing

using Turing: DynamicPPL
import Distributions: logpdf, params

export gev_logpdf, gp_logpdf, GEVKernel, GPKernel
export design_matrix
export EVAFit, params, getdistribution, returnlevel, loglike
export gevfit, gpfit, gevfitbayes, gpfitbayes, gevfit_lmom
export Station, WaterLevelRecord, AnnMaxRecord
export load_water_level, waterlevels, obstimes, obsyears, baseline
export plotting_positions, return_period_axis, return_period_axis!
export decluster, decluster_daily, mrl_data, stability_data, pot_return_level
export mrl_plot, stability_plot

include("kernels.jl")
include("models.jl")
include("fitting.jl")
include("lmoments.jl")
include("tables.jl")
include("tidegauge.jl")   # defines DETREND_METHODS, which records.jl interpolates
include("records.jl")
include("pot.jl")

"""
    plotting_positions(y) -> (sorted, p, T)

Sort `y` from largest to smallest and return the Weibull plotting positions
`p = i / (n + 1)` beside the return periods `T = 1 / p`.

Dividing by `n + 1` rather than `n` keeps every estimate strictly inside
`(0, 1)`, so the smallest observation does not land at `p = 1`.
"""
function plotting_positions(y)
    sorted = sort(collect(y); rev=true)
    n = length(sorted)
    p = (1:n) ./ (n + 1)
    return sorted, p, 1 ./ p
end

# Defined here and given methods by the Makie extension, so that loading a
# backend is what turns them on and a lab that only fits pays nothing for Makie.
function return_period_axis! end
function return_period_axis end
function mrl_plot end
function stability_plot end

end # module
