# Code review fixes — Implementation Plan

**Status:** done 2026-09-23. All 12 tasks landed; suite passes. Not done: JuliaFormatter is not installed in any environment on this machine, so Task 12 Step 1 was skipped. Network testset not run.

**Goal:** Fix the eight defects, the compat gap, the coverage gaps, and the doc drift found in the 2026-09-23 review, with a regression test for each fix.
**Architecture:** Every fix is local to the function named. The one design change is that covariates are standardized inside the fit (centre and scale stored on the fit), the model works on the standardized scale, and `params` back-transforms so users still read a slope per unit of their covariate. `getdistribution` and `posterior_distributions` keep using the standardized coefficients against the standardized design matrix, so their output is unchanged.
**Spec:** the review thread of 2026-09-23 (no written spec). Decisions taken there: analytic delta-method standard error for σ*; cache filename carries datum and year range; standardize covariates and back-transform in `params`; Julia 1.12 is required.

Every task ends by running the full suite. The command, used throughout, is:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run JuliaFormatter before the final commit, from the repo root:

```sh
julia -e 'using JuliaFormatter; format(".")'
```

## Task 1: Correct the standard error of the modified scale

**Files:**
- Modify: `src/pot.jl`
- Modify: `test/runtests.jl`

- [x] **Step 1:** In `src/pot.jl`, replace line 144

  ```julia
              sigma_star_se[k] = sigma / sqrt(n)
  ```

  with

  ```julia
              # Delta method on σ* = σ - ξu with the GPD asymptotic covariance
              # (1+ξ)/n [2σ², -σ; -σ, 1+ξ] (Coles 2001, §4.3.4). Checked against
              # simulated MLEs on 2026-09-23: the σ,ξ covariance is negative.
              sigma_star_se[k] = sqrt((1 + xi) * (2sigma^2 + 2sigma * u + u^2 * (1 + xi)) / n)
  ```

- [x] **Step 2:** In the `stability_data` docstring (`src/pot.jl:109-119`), replace "with approximate standard errors" with "with delta-method standard errors from the GPD asymptotic covariance, valid for ξ > -0.5".

- [x] **Step 3:** In `test/runtests.jl`, replace the `stability_data` testset (lines 450-457) with

  ```julia
  @testset "stability_data" begin
      rng = MersenneTwister(543)
      data = rand(rng, GeneralizedPareto(0.0, 2.0, 0.1), 10_000)
      t, ss, xi, ss_se, xi_se = stability_data(data; n_thresholds=20)
      ok = .!isnan.(xi)
      @test any(ok)
      @test mean(xi[ok]) ≈ 0.1 atol = 0.15
      # σ* = σ_u - ξu is constant in u when the data are GPD above every threshold
      @test all(≈(2.0; atol=0.3), ss[ok])
      # the band widens with the threshold, since σ* inherits u² Var(ξ)
      @test ss_se[ok][end] > ss_se[ok][1]
      # against a simulated sampling distribution of σ* at one threshold
      u, n = 2.0, 200
      draws = [
          let (s, x) = CEVE543Utils._gpd_mle_quick(rand(rng, GeneralizedPareto(0.0, 0.2, 0.1), n))
              s - x * u
          end for _ in 1:500
      ]
      expected = sqrt(1.1 * (2 * 0.2^2 + 2 * 0.2 * u + u^2 * 1.1) / n)
      @test std(draws) ≈ expected rtol = 0.25
  end
  ```

- [x] **Step 4:** Run the suite. Expect the new `stability_data` assertions to pass and nothing else to change.

## Task 2: Key the download cache on datum and year range

**Files:**
- Modify: `src/tidegauge.jl`
- Modify: `test/runtests.jl`

- [x] **Step 1:** Reorder the keyword arguments of `load_water_level` (`src/tidegauge.jl:47-54`) so `cache` comes last and its default sees the other keywords:

  ```julia
  function load_water_level(
      station::AbstractString;
      first_year::Integer=1928,
      last_year::Integer=year(today()),
      datum::AbstractString="MSL",
      name::AbstractString="",
      cache=default_cache(station, datum, first_year, last_year),
  )
  ```

- [x] **Step 2:** Replace line 91

  ```julia
  default_cache(station::AbstractString) = "$station-hourly.csv"
  ```

  with

  ```julia
  default_cache(station, datum, first_year, last_year) =
      "$station-$datum-$first_year-$last_year-hourly.csv"
  ```

- [x] **Step 3:** In the `load_water_level` docstring, replace the `cache` bullet's `Defaults to "<station>-hourly.csv"` with `Defaults to "<station>-<datum>-<first_year>-<last_year>-hourly.csv", so a different datum or period downloads again instead of relabelling the old file`.

- [x] **Step 4:** Add to `test/runtests.jl`, after the "load_water_level reads a cache without the network" testset:

  ```julia
  @testset "default cache name carries datum and years" begin
      @test CEVE543Utils.default_cache("8638610", "NAVD88", 1990, 2000) ==
          "8638610-NAVD88-1990-2000-hourly.csv"
  end
  ```

- [x] **Step 5:** Run the suite.

## Task 3: Never cache a partial download, and retry only network errors

**Files:**
- Modify: `src/tidegauge.jl`

- [x] **Step 1:** Change the import on `src/tidegauge.jl:11` to

  ```julia
  using Downloads: download, RequestError
  ```

- [x] **Step 2:** Make `append_year!` report whether the year's download succeeded. Replace lines 113-123 with

  ```julia
      body = nothing
      for attempt in 1:3
          try
              body = sprint(io -> download(url, io))
              break
          catch err
              # Only a failed request is worth retrying; Ctrl-C and programming
              # errors propagate.
              err isa RequestError || rethrow()
              attempt < 3 && (sleep(2^attempt); continue)
              @warn "station $station year $yr: download failed after 3 attempts, skipping" err
              return false
          end
      end
  ```

  and change the two `return nothing` at the old lines 129 and 139 to `return true`. A year the gauge did not report is a complete answer, so it counts as success.

- [x] **Step 3:** Update the `append_year!` docstring (lines 95-100) to

  ```julia
  """
      append_year!(times, levels, station, yr, datum) -> Bool

  Download one year of hourly readings and append them. Returns `true` when the
  request succeeded, including for a year the gauge did not report, which adds
  nothing since a long record routinely has gaps. Returns `false` when the
  download itself failed three times.
  """
  ```

- [x] **Step 4:** In `load_water_level`, replace lines 72-85 with

  ```julia
      times, levels = DateTime[], Float64[]
      failed = Int[]
      for yr in first_year:last_year
          append_year!(times, levels, station, yr, datum) || push!(failed, yr)
          # Three years failing in a row means the network, not the gauge, is the
          # problem, and waiting out the rest of the century helps nobody.
          if length(failed) >= 3 && failed[(end - 2):end] == [yr - 2, yr - 1, yr]
              throw(
                  ArgumentError(
                      "downloads for $station failed for $(failed[end - 2])-$yr; " *
                      "check the network connection and try again",
                  ),
              )
          end
      end
      isempty(levels) && throw(
          ArgumentError(
              "station $station returned no readings for $first_year-$last_year; " *
              "check the id at $COOPS_STATIONS",
          ),
      )

      order = sortperm(times)
      times, levels = times[order], levels[order]
      if cache !== nothing
          if isempty(failed)
              CSV.write(cache, (time=times, level_m=levels))
          else
              @warn "not caching $station: the download skipped $(length(failed)) year(s)" failed
          end
      end
  ```

- [x] **Step 5:** In the `load_water_level` docstring, after "It is cached, so that cost is paid once.", add the sentence: "A download that skipped a year after repeated failures is not cached, so running it again fills the gap."

- [x] **Step 6:** Run the suite. The network testset is skipped by default; run it once with `CEVE543_NETWORK_TESTS=1` if the network is available, and expect the Sewells Point assertions to pass unchanged.

## Task 4: Standardize covariates inside the fit, back-transform in `params`

**Files:**
- Modify: `src/models.jl`
- Modify: `src/fitting.jl`
- Modify: `test/runtests.jl`

- [x] **Step 1:** In `src/models.jl`, after `design_matrix` (after line 23), add

  ```julia
  """
      standardize(X) -> (X, center, scale)

  Centre and scale every column of a design matrix except the intercept, so the
  optimizer and the sampler see coefficients of order one whatever the units of
  the covariate. `center` and `scale` hold one entry per covariate column and
  are what [`params`](@ref) uses to report coefficients on the original scale.

  A constant column keeps a scale of one, since dividing by zero would help
  nobody and the column is collinear with the intercept anyway.
  """
  function standardize(X::AbstractMatrix)
      center = vec(mean(X[:, 2:end]; dims=1))
      scale = vec(std(X[:, 2:end]; dims=1))
      scale[scale .== 0] .= 1
      Z = copy(X)
      Z[:, 2:end] .= (X[:, 2:end] .- center') ./ scale'
      return (X=Z, center=center, scale=scale)
  end
  ```

- [x] **Step 2:** In `src/fitting.jl`, change the `EVAFit` docstring line 6 from "`Xs` holds the design matrix used for each parameter" to "`Xs` holds, for each parameter, the standardized design matrix as `X` with its `center` and `scale`".

- [x] **Step 3:** Replace `params` (`src/fitting.jl:19-36`) with

  ```julia
  """
      params(fit) -> NamedTuple

  The fitted coefficients for each parameter, intercept first, on the scale of
  the covariates as they were passed in.

  The fit itself works on centred and scaled covariates, so `fit.estimate`
  holds coefficients on that standardized scale. This function undoes the
  transform: a slope per year comes back as a slope per year.

  Defined for the point estimates. A fit from [`gevfitbayes`](@ref) or
  [`gpfitbayes`](@ref) holds a whole posterior rather than one value per
  coefficient; use [`posterior_distributions`](@ref), which applies the same
  design matrix to every draw.
  """
  function params(fit::EVAFit)
      raw = _raw_params(fit)
      return NamedTuple{keys(raw)}(
          Tuple(_original_scale(raw[k], fit.Xs[_design_key(k)]) for k in keys(raw))
      )
  end

  # The coefficients as the optimizer left them, on the standardized scale.
  function _raw_params(fit::EVAFit)
      fit.estimate isa Turing.Optimisation.ModeResult || throw(
          ArgumentError(
              "this fit holds a posterior sample rather than a point estimate; " *
              "use `posterior_distributions`",
          ),
      )
      return NamedTuple(fit.estimate.params)
  end

  # `β_location` was fitted against `Xs.location`, and so on.
  _design_key(name::Symbol) = Symbol(string(name)[(length("β_") + 1):end])

  # Coefficients on the original covariate scale. With z = (x - c) / s the linear
  # predictor β₀ + Σ βⱼ zⱼ equals (β₀ - Σ βⱼ cⱼ / sⱼ) + Σ (βⱼ / sⱼ) xⱼ.
  function _original_scale(β, design)
      slopes = β[2:end] ./ design.scale
      return vcat(β[1] - sum(slopes .* design.center), slopes)
  end
  ```

- [x] **Step 4:** Update the remaining uses of `Xs` in `src/fitting.jl`:

  Line 38:

  ```julia
  _stationary(fit::EVAFit) = all(d -> size(d.X, 2) == 1, values(fit.Xs))
  ```

  Line 50:

  ```julia
  getdistribution(fit::EVAFit) = _distributions(fit, _raw_params(fit))
  ```

  In `_distributions` (lines 83-101), change every `fit.Xs.location`, `fit.Xs.logscale`, `fit.Xs.shape` to `fit.Xs.location.X`, `fit.Xs.logscale.X`, `fit.Xs.shape.X`.

  Lines 158 and 160-164:

  ```julia
  _zeros(design) = zeros(size(design.X, 2))

  function _shape_start(design)
      β = zeros(size(design.X, 2))
      β[1] = SHAPE_INIT
      return β
  end
  ```

  Lines 180-186:

  ```julia
  function _gev_matrices(n, loc, logscale, shape)
      return (
          location=standardize(design_matrix(n, loc)),
          logscale=standardize(design_matrix(n, logscale)),
          shape=standardize(design_matrix(n, shape)),
      )
  end
  ```

  Lines 193-196:

  ```julia
      Xs = (
          logscale=standardize(design_matrix(length(excess), sub(logscalecov))),
          shape=standardize(design_matrix(length(excess), sub(shapecov))),
      )
  ```

  In `gevfit`, `gpfit`, `gevfitbayes`, `gpfitbayes`, change the model construction to pass the matrices:

  ```julia
      model = gev_model(y, Xs.location.X, Xs.logscale.X, Xs.shape.X, prior_scale)
  ```

  ```julia
      model = gp_model(excess, Xs.logscale.X, Xs.shape.X, prior_scale)
  ```

- [x] **Step 5:** In the `gevfitbayes` docstring (`src/fitting.jl:264-266`), replace the `prior_scale` paragraph with

  ```
  Covariates are centred and scaled before sampling, so `prior_scale`, the prior
  variance of every location and log-scale coefficient, applies to a slope per
  standard deviation of the covariate; the default of 100 is a standard deviation
  of 10 on that scale. The shape coefficients have prior standard deviation 0.5.
  The chain in `fit.estimate` is on the standardized scale;
  [`posterior_distributions`](@ref) accounts for that.
  ```

- [x] **Step 6:** In `test/runtests.jl`, add after the "gevfit, location trend" testset:

  ```julia
  @testset "gevfit with covariates on their raw scale" begin
      # Calendar years are what a student passes, and the intercept at year zero
      # is then far from any prior or starting value. The fit standardizes
      # internally and reports back on the original scale.
      rng = MersenneTwister(543)
      n, slope = 4_000, 0.02
      years = collect(range(1900, 2000; length=n))
      y = [rand(rng, GeneralizedExtremeValue(4.0 + slope * (yr - 1900), 0.8, 0.1)) for yr in years]

      fit = gevfit(y; locationcov=[years])
      β = params(fit).β_location
      @test β[2] ≈ slope rtol = 0.20                          # per year, as passed in
      @test β[1] + β[2] * 1900 ≈ 4.0 atol = 0.2               # the line passes through the data
      dists = getdistribution(fit)
      @test dists[1].μ ≈ 4.0 atol = 0.2
      @test dists[end].μ ≈ 6.0 atol = 0.2

      # the same fit, centred by hand, gives the same distributions
      by_hand = gevfit(y; locationcov=[years .- 1950])
      @test getdistribution(by_hand)[end].μ ≈ dists[end].μ atol = 1e-3
      @test params(by_hand).β_location[2] ≈ β[2] atol = 1e-4
  end
  ```

- [x] **Step 7:** Replace the "posterior_distributions with covariates" testset (lines 197-210) with the same test on raw years:

  ```julia
  @testset "posterior_distributions with covariates" begin
      rng = MersenneTwister(543)
      n = 300
      # Raw calendar years: the fit centres and scales them, so NUTS converges
      # without the user having to.
      years = collect(range(1900, 2000; length=n))
      y = [rand(rng, GeneralizedExtremeValue(4.0 + 0.01 * (yr - 1900), 0.8, 0.1)) for yr in years]
      fit = gevfitbayes(y; locationcov=[years], n_samples=200, n_chains=2, rng=rng)

      draws = posterior_distributions(fit)
      @test length(draws) == 200 * 2
      @test all(d -> length(d) == n, draws)      # one distribution per row, per draw
      # the location climbs across the record by the slope times the covariate's span
      @test median(last(d).μ - first(d).μ for d in draws) ≈ 1.0 rtol = 0.3
      @test median(first(d).μ for d in draws) ≈ 4.0 rtol = 0.1
  end
  ```

- [x] **Step 8:** Run the suite. The existing trend tests (`t` in [0, 1]) must still pass, which checks that the back-transform is the identity up to floating point when the covariate is already of order one.

## Task 5: Accept matrix covariates in `gpfit`

**Files:**
- Modify: `src/fitting.jl`
- Modify: `test/runtests.jl`

- [x] **Step 1:** In `_exceedances` (`src/fitting.jl:192`), replace

  ```julia
      sub(c) = isnothing(c) ? nothing : [collect(v)[keep] for v in c]
  ```

  with

  ```julia
      # A matrix iterates element by element, so take its columns first, the way
      # `design_matrix` does.
      sub(c) = isnothing(c) ? nothing : [collect(v)[keep] for v in _covariate_columns(c)]
  ```

  and after `_exceedances` add

  ```julia
  _covariate_columns(c::AbstractMatrix) = eachcol(c)
  _covariate_columns(c) = c
  ```

- [x] **Step 2:** Add to `test/runtests.jl`, after the "gpfit keeps only the exceedances" testset:

  ```julia
  @testset "gpfit with covariates" begin
      rng = MersenneTwister(543)
      n, threshold = 4_000, 10.0
      x = collect(range(0, 1; length=n))
      # the log-scale climbs by 0.5 over the record
      y = [rand(rng, GeneralizedPareto(threshold, exp(log(2.0) + 0.5 * xᵢ), 0.1)) for xᵢ in x]

      fit = gpfit(y, threshold; logscalecov=[x])
      @test params(fit).β_logscale[2] ≈ 0.5 rtol = 0.3
      @test length(getdistribution(fit)) == n

      # a matrix with one column per covariate is the same fit
      as_matrix = gpfit(y, threshold; logscalecov=reshape(x, :, 1))
      @test params(as_matrix).β_logscale ≈ params(fit).β_logscale atol = 1e-6

      # covariates are subset to the exceedances alongside the data
      mixed = vcat(y, fill(5.0, 100))
      x_mixed = vcat(x, zeros(100))
      @test length(gpfit(mixed, threshold; logscalecov=[x_mixed]).y) == n
  end
  ```

- [x] **Step 3:** Run the suite.

## Task 6: Refuse a `:linear` detrend with fewer than two years

**Files:**
- Modify: `src/tidegauge.jl`
- Modify: `test/runtests.jl`

- [x] **Step 1:** In `detrend_baseline` (`src/tidegauge.jl:168-170`), replace

  ```julia
      if method === :linear
          slope, intercept = ols(years, annmax)
  ```

  with

  ```julia
      if method === :linear
          length(years) < 2 && throw(
              ArgumentError(
                  "linear detrending needs at least two years and got " *
                  "$(length(years)); use detrend=:none or :msl for a short record",
              ),
          )
          slope, intercept = ols(years, annmax)
  ```

- [x] **Step 2:** In the "detrend_baseline and recentre" testset, after the `:quadratic` assertion (line 249), add

  ```julia
      # one point defines no line, and a NaN baseline would be silent
      @test_throws ArgumentError detrend_baseline([2000], [5.0], [0.0], :linear)
      @test detrend_baseline([2000], [5.0], [0.0], :msl) == [0.0]
  ```

- [x] **Step 3:** Run the suite.

## Task 7: Guard `gevfit_lmom` and test it

**Files:**
- Modify: `src/lmoments.jl`
- Modify: `test/runtests.jl`

- [x] **Step 1:** Replace `src/lmoments.jl:8-49` with

  ```julia
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

  Algorithm: Hosking et al. (1985), whose polynomial approximation for the shape
  holds for `-0.5 <= ξ <= 0.5`. Needs at least three observations, since the
  third probability weighted moment is undefined below that.
  """
  function gevfit_lmom(y::AbstractVector{<:Real})
      length(y) >= 3 || throw(
          ArgumentError("gevfit_lmom needs at least 3 observations, got $(length(y))")
      )
      b0 = pwm(y, 1, 0, 0)
      b1 = pwm(y, 1, 1, 0)
      b2 = pwm(y, 1, 2, 0)

      c = (2b1 - b0) / (3b2 - b0) - log(2) / log(3)
      k = 7.8590c + 2.9554c^2
      if abs(k) < 1e-8
          # The Gumbel limit, where k drops out of the scale and location.
          σ = (2b1 - b0) / log(2)
          return GeneralizedExtremeValue(b0 - eulergamma * σ, σ, 0.0)
      end
      σ = k * (2b1 - b0) / ((1 - 2^(-k)) * gamma(1 + k))
      μ = b0 - σ / k * (1 - gamma(1 + k))
      return GeneralizedExtremeValue(μ, σ, -k)
  end
  ```

- [x] **Step 2:** Add to `test/runtests.jl`, after the "gevfit, stationary" testset:

  ```julia
  @testset "gevfit_lmom" begin
      truth = GeneralizedExtremeValue(4.0, 0.8, 0.15)
      y = rand(MersenneTwister(543), truth, 20_000)
      d = gevfit_lmom(y)
      @test d.μ ≈ truth.μ rtol = 0.02
      @test d.σ ≈ truth.σ rtol = 0.05
      @test d.ξ ≈ truth.ξ atol = 0.03

      # the Gumbel case comes back finite
      g = gevfit_lmom(rand(MersenneTwister(543), Gumbel(4.0, 0.8), 20_000))
      @test isfinite(g.σ) && isfinite(g.μ)
      @test g.ξ ≈ 0.0 atol = 0.03

      # probability weighted moments: b0 is the mean
      @test CEVE543Utils.pwm([1.0, 2.0, 3.0], 1, 0, 0) ≈ 2.0

      @test_throws ArgumentError gevfit_lmom([1.0, 2.0])
  end
  ```

- [x] **Step 3:** Run the suite.

## Task 8: Small correctness fixes

**Files:**
- Modify: `src/kernels.jl`
- Modify: `src/fitting.jl`
- Modify: `src/pot.jl`
- Modify: `test/runtests.jl`

- [x] **Step 1:** In `gp_logpdf` (`src/kernels.jl:71-77`), after `z = y / σ` add

  ```julia
      z < 0 && return oftype(z, -Inf)   # an exceedance is measured upward from the threshold
  ```

  and in the "log densities" testset add

  ```julia
      @test gp_logpdf(-1.0, 2.0, 0.2) == -Inf     # below the threshold is outside the support
  ```

- [x] **Step 2:** Replace `loglike` (`src/fitting.jl:142-147`) with

  ```julia
  """
      loglike(fit) -> Real

  The log likelihood at the point estimate. Defined for [`gevfit`](@ref) and
  [`gpfit`](@ref); a posterior sample has no single value.
  """
  function loglike(fit::EVAFit)
      fit.estimate isa Turing.Optimisation.ModeResult || throw(
          ArgumentError("loglike is defined for a point estimate; this fit holds a posterior sample")
      )
      return fit.estimate.lp
  end
  ```

  and in the "gevfitbayes" testset, next to `@test_throws ArgumentError params(fit)`, add

  ```julia
      @test_throws ArgumentError loglike(fit)
  ```

- [x] **Step 3:** Remove `prior_scale` from the point-estimate fitters. In `gevfit` (`src/fitting.jl:214-227`) delete the `prior_scale=100.0,` keyword and change the model line to

  ```julia
      # The prior is irrelevant to maximum likelihood; the model takes a value anyway.
      model = gev_model(y, Xs.location.X, Xs.logscale.X, Xs.shape.X, 1.0)
  ```

  Do the same in `gpfit` (`src/fitting.jl:240-253`):

  ```julia
      model = gp_model(excess, Xs.logscale.X, Xs.shape.X, 1.0)
  ```

  Remove `prior_scale` from both signature lines in the docstrings (`src/fitting.jl:201` and `:230`). In the `gevfitbayes` docstring, lines 260-261, change "The model, the covariate keywords, and the priors are the ones `gevfit` uses" to "The model and the covariate keywords are the ones [`gevfit`](@ref) uses, with the priors below on top".

  In the "gevfit, stationary" testset, define `y = rand(MersenneTwister(543), truth, 5_000)` on its own line, use it in the `gevfit` call, and add

  ```julia
      @test_throws MethodError gevfit(y; prior_scale=1.0)   # maximum likelihood has no prior
  ```

- [x] **Step 4:** In `decluster` (`src/pot.jl:25`), before `exceed_idx = ...`, add

  ```julia
      issorted(times) || throw(ArgumentError("times must be in increasing order for the gaps to mean anything"))
  ```

  and in the "decluster" testset add

  ```julia
      @test_throws ArgumentError decluster(reverse(times), levels, 5.0)
      # a gap exactly equal to the minimum does not start a new cluster
      two = [DateTime(2000, 1, 1, 0), DateTime(2000, 1, 1, 6)]
      @test length(decluster(two, [10.0, 9.0], 5.0; min_gap_hours=6)[1]) == 1
  ```

- [x] **Step 5:** Run the suite.

## Task 9: State the Julia requirement

**Files:**
- Modify: `Project.toml` (through Pkg)

- [x] **Step 1:** Run

  ```sh
  julia --project=. -e 'using Pkg; Pkg.compat("julia", "1.12")'
  ```

- [x] **Step 2:** Confirm `Project.toml` `[compat]` now has `julia = "1.12"` and nothing else moved:

  ```sh
  git diff Project.toml
  ```

## Task 10: Close the coverage gaps

**Files:**
- Modify: `test/runtests.jl`

- [x] **Step 1:** Extend the "mrl_data" testset (lines 437-448) with the bounded grid and the count floor:

  ```julia
      # an explicit grid and count floor
      t2, m2, _, _ = mrl_data(data; n_thresholds=5, lo=1.0, hi=9.0, min_count=50)
      @test t2 == collect(range(1.0, 9.0; length=5))
      @test isfinite(m2[1])
      @test isnan(mrl_data(data; n_thresholds=3, lo=1.0, hi=40.0, min_count=50)[2][end])
  ```

  and replace the tautological bounds check on line 445 with

  ```julia
      @test all(hi[ok] .- lo[ok] .> 0)
      @test hi[ok][1] - lo[ok][1] < hi[ok][end] - lo[ok][end]   # fewer exceedances, wider band
  ```

- [x] **Step 2:** Add a `gevfit` test with a log-scale covariate, after "gevfit with covariates on their raw scale":

  ```julia
  @testset "gevfit, scale trend" begin
      rng = MersenneTwister(543)
      n = 4_000
      x = collect(range(0, 1; length=n))
      y = [rand(rng, GeneralizedExtremeValue(4.0, exp(log(0.8) + 0.5 * xᵢ), 0.1)) for xᵢ in x]
      fit = gevfit(y; logscalecov=[x])
      @test params(fit).β_logscale[2] ≈ 0.5 rtol = 0.3
      @test length(params(fit).β_location) == 1
      dists = getdistribution(fit)
      @test dists[end].σ > dists[1].σ
  end
  ```

- [x] **Step 3:** Add smoke tests for the two plot recipes, after the "return_period_axis" testset:

  ```julia
  @testset "mrl_plot and stability_plot build a figure" begin
      rng = MersenneTwister(543)
      data = rand(rng, GeneralizedPareto(0.0, 2.0, 0.1), 3_000)
      fig = mrl_plot(data; n_thresholds=20)
      @test fig isa Figure
      @test length(contents(fig[1, 1])) == 1
      fig2 = stability_plot(data; n_thresholds=10)
      @test fig2 isa Figure
      @test length(contents(fig2[2, 1])) == 1     # two axes stacked
  end
  ```

- [x] **Step 4:** Add a smoke test for the table methods not yet exercised, after "gevfit from a table":

  ```julia
  @testset "gpfit and the Bayesian fitters from a table" begin
      rng = MersenneTwister(543)
      df = DataFrame(; x=collect(range(0, 1; length=1_000)))
      df.q = rand(rng, GeneralizedPareto(10.0, 2.0, 0.1), 1_000)
      fit = gpfit(df, :q, 10.0; logscalecovid=[:x])
      @test length(params(fit).β_logscale) == 2
      bayes = gpfitbayes(df, :q, 10.0; n_samples=50, n_chains=1, rng=rng)
      @test length(posterior_distributions(bayes)) == 50
  end
  ```

- [x] **Step 5:** Run the suite.

## Task 11: Bring the docs in line with the code

**Files:**
- Modify: `README.md`
- Modify: `src/CEVE543Utils.jl`
- Modify: `src/fitting.jl`
- Modify: `src/pot.jl`
- Modify: `src/tidegauge.jl`
- Modify: `src/records.jl`
- Modify: `ext/CEVE543UtilsMakieExt.jl`

- [x] **Step 1:** In `README.md`, replace the last paragraph (lines 43-44) with

  ````markdown
  ## Tide gauges and peaks over threshold

  `load_water_level` downloads a NOAA CO-OPS station's hourly record, `AnnMaxRecord` takes its annual maxima and `detrend` removes the sea-level trend from every reading.
  Annual blocks are UTC calendar years.

  ```julia
  record = load_water_level("8638610")                 # Sewells Point, VA, cached locally
  annmax = AnnMaxRecord(record; detrend=:msl)          # one maximum per full year
  peaks, excess = decluster(obstimes(record), record.levels, 1.0)   # clusters 72 h apart
  mrl_data(record.levels)                              # mean residual life against threshold
  stability_data(record.levels)                        # GPD parameters against threshold
  ```

  Makie is a weak dependency.
  `plotting_positions` is always available.
  `return_period_axis!`, `return_period_axis`, `mrl_plot` and `stability_plot` appear once you load a backend.
  ````

- [x] **Step 2:** In the module docstring (`src/CEVE543Utils.jl:27-28`), replace

  ```
  Plot helpers live in an extension: `plotting_positions` is always available, and
  `return_period_axis!` appears once Makie is loaded.
  ```

  with

  ```
  Peaks over threshold diagnostics are `decluster`, `mrl_data`, `stability_data`
  and `pot_return_level`. Plot helpers live in an extension: `plotting_positions`
  is always available, and `return_period_axis!`, `mrl_plot` and `stability_plot`
  appear once Makie is loaded.
  ```

- [x] **Step 3:** In `src/fitting.jl`, replace the stale comment at lines 152-155 with

  ```julia
  # The shape is the exception: it starts a little away from zero. The densities
  # keep a derivative in ξ through zero (see `log1p_over`), but the likelihood is
  # flat enough there that an optimizer started exactly at zero can stop early.
  ```

- [x] **Step 4:** In `getdistribution`'s docstring (`src/fitting.jl:43`), change "One distribution per observation, in the order the data came in." to "One distribution per observation, in the order the data came in; for a peaks over threshold fit, one per exceedance."

- [x] **Step 5:** Replace the `mrl_data` docstring (`src/pot.jl:68-76`) with

  ```julia
  """
      mrl_data(values; n_thresholds=80, lo_quantile=0.5, hi_quantile=0.995, lo, hi, min_count=5)

  Compute the mean residual life at a grid of thresholds.

  The grid runs from the `lo_quantile` to the `hi_quantile` of `values`, or from
  `lo` to `hi` when those are given. Returns `(thresholds, means, lower, upper)`
  where `lower` and `upper` are pointwise 95% confidence bounds. Thresholds with
  fewer than `min_count` exceedances produce `NaN`.
  """
  ```

- [x] **Step 6:** In the `load_water_level` docstring (`src/tidegauge.jl:25-26`), change "in metres above the requested datum." to "in metres above the requested datum, timestamped in UTC." In the `AnnMaxRecord` docstring in `src/records.jl`, after the sentence ending "which is what the raw record makes available." (line 221), add "Years are UTC calendar years, as the readings are timestamped."

- [x] **Step 7:** In `ext/CEVE543UtilsMakieExt.jl:4-5`, delete line 4 (`using Makie: Axis, Figure, band!, lines!, scatter!, xlims!`), since line 5 imports all of them.

- [x] **Step 8:** Run the suite, then check the README for blank lines around the new heading and code block.

## Task 12: Format, verify, bump the version

**Files:**
- Modify: `Project.toml`
- Modify: every file touched above (formatter)

- [x] **Step 1:** Run JuliaFormatter from the repo root (command at the top of this plan) and confirm `git diff --stat` shows only files this plan touched.

- [x] **Step 2:** Run the suite one final time and confirm every testset reports zero failures.

- [x] **Step 3:** Bump `version = "0.5.0"` to `version = "0.6.0"` in `Project.toml`. Removing `prior_scale` from `gevfit` and `gpfit` and renaming the default cache file are both changes a caller can notice.

- [x] **Step 4:** Do not commit until the user says so.
