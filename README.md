# CEVE543Utils

Helpers for [CEVE 543: Statistical-Physical Methods for Hydroclimate Extremes and
Catastrophes](https://ceve543.github.io/).

Extreme value fitting with the interface of
[Extremes.jl](https://github.com/jojal5/Extremes.jl) over a Turing model, so
`gevfit`, `gpfit`, `getdistribution` and `returnlevel` mean what they mean
there, and one model serves both the point estimate and the posterior.

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
