# CEVE543Utils

Extreme value fitting for CEVE 543, with the interface of
[Extremes.jl](https://github.com/jojal5/Extremes.jl) over a Turing model.

`gevfit`, `gpfit`, `getdistribution` and `returnlevel` mean what they mean in
Extremes.jl, and covariates are named the same way. What differs is underneath:
each family is one Turing model, entered either by mode estimation for a point
estimate or by NUTS for the posterior, so covariates reach both.

```julia
fit = gevfit(annmax)                       # stationary, maximum likelihood
gev = only(getdistribution(fit))           # one distribution
returnlevel(fit, 100)                      # the 100-year level

fit = gevfit(df, :lsl; locationcovid=[:year])   # location moves with the year
getdistribution(fit)                            # one distribution per row
```

Peaks over threshold works the same way, with the threshold as a second
argument:

```julia
fit = gpfit(discharge, 500.0; logscalecovid=[:enso])
```

## Installing

Not registered. Add it by URL, and record that URL in `[sources]` so the project
resolves without a manifest:

```toml
[deps]
CEVE543Utils = "9cb47f62-1030-411a-9180-bf3a9ffb897b"

[sources]
CEVE543Utils = {url = "https://github.com/CEVE543/CEVE543Utils"}
```

## Dependencies

Deliberately thin. Tables.jl rather than DataFrames, so a DataFrame, a
`NamedTuple` of vectors, and a `CSV.File` all work without the dependency. Makie
is a weak dependency: `plotting_positions` is always there, and
`return_period_axis!` appears once a backend is loaded.
