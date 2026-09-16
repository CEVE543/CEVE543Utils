# CEVE543Utils

Helpers for [CEVE 543: Statistical-Physical Methods for Hydroclimate Extremes and
Catastrophes](https://ceve543.github.io/).

Extreme value fitting inspired by
[Extremes.jl](https://github.com/jojal5/Extremes.jl) (MIT, Jalbert, Farmer &
Roy 2020), with a lighter dependency footprint built on Turing.
`gevfit`, `gpfit`, `getdistribution` and `returnlevel` follow the Extremes.jl
interface, and one Turing model serves both the point estimate and the
posterior.
The L-moment estimator (`gevfit_lmom`) adapts the probability weighted moment
algorithm from Extremes.jl, following Hosking, Wallis & Wood (1985).

```julia
fit = gevfit(annmax)                            # stationary, maximum likelihood
only(getdistribution(fit))                      # the fitted distribution
returnlevel(fit, 100)                           # the 100-year level

fit = gevfit(df, :lsl; locationcovid=[:year])   # location moves with the year
getdistribution(fit)                            # one distribution per row

fit = gpfit(discharge, 500.0)                   # peaks over a threshold
fit = gevfitbayes(annmax)                       # posterior instead of a point
```

## Installing

Not registered. Add it by URL and record that URL in `[sources]`, so the project
resolves without a manifest:

```toml
[deps]
CEVE543Utils = "9cb47f62-1030-411a-9180-bf3a9ffb897b"

[sources]
CEVE543Utils = {url = "https://github.com/CEVE543/CEVE543Utils"}
```

Makie is a weak dependency: `plotting_positions` is always available, and
`return_period_axis!` appears once you load a backend.
