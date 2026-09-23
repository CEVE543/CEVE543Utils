using CairoMakie          # loading a backend is what turns the Makie extension on
using CEVE543Utils
using CEVE543Utils: detrend_baseline, recentre
using CSV
using DataFrames
using Dates
using Distributions
using Random
using Statistics
using Test
using Unitful: @u_str, unit, ustrip

@testset "plotting_positions" begin
    sorted, p, T = plotting_positions([1.0, 5.0, 3.0])
    @test sorted == [5.0, 3.0, 1.0]
    @test p ≈ [1 / 4, 2 / 4, 3 / 4]
    @test T ≈ 1 ./ p
    @test maximum(T) ≈ length(sorted) + 1     # the record reaches n + 1 and no further
    @test all(0 .< p .< 1)                    # the n + 1 is what keeps this true
end

@testset "log densities" begin
    # against Distributions.jl, where both are also defined
    @test gev_logpdf(5.0, 4.0, 0.8, 0.15) ≈
        logpdf(GeneralizedExtremeValue(4.0, 0.8, 0.15), 5.0)
    @test gp_logpdf(3.0, 2.0, 0.2) ≈ logpdf(GeneralizedPareto(0.0, 2.0, 0.2), 3.0)

    # the ξ → 0 branch is the Gumbel and the exponential
    @test gev_logpdf(5.0, 4.0, 0.8, 0.0) ≈ logpdf(Gumbel(4.0, 0.8), 5.0)
    @test gp_logpdf(3.0, 2.0, 0.0) ≈ logpdf(Exponential(2.0), 3.0)

    # outside the support, or at a nonsense scale, it declines rather than throws
    @test gev_logpdf(-100.0, 4.0, 0.8, 0.5) == -Inf
    @test gp_logpdf(1.0, -1.0, 0.2) == -Inf
    @test gp_logpdf(100.0, 1.0, -0.5) == -Inf

    # `log1p` raises on an argument at or below -1 rather than returning a
    # non-finite value, so the support test has to come first.
    @test gev_logpdf(4.0 - 0.8 / 0.5, 4.0, 0.8, 0.5) == -Inf   # exactly at the boundary
    @test gp_logpdf(2.0, 1.0, -0.5) == -Inf
    @test gp_logpdf(-1.0, 2.0, 0.2) == -Inf     # below the threshold is outside the support
end

@testset "the shape carries a derivative through zero" begin
    # The density is written through log1p(ξz)/ξ, whose series expansion keeps ξ
    # in the expression at the Gumbel limit. Branching onto the Gumbel instead
    # drops ξ, and the derivative of an expression that no longer contains a
    # parameter is zero, which an optimizer reads as convergence.
    slope(ξ; h=1e-9) =
        (gev_logpdf(5.0, 4.0, 0.8, ξ + h) - gev_logpdf(5.0, 4.0, 0.8, ξ - h)) / 2h

    for ξ in (1e-8, 0.0, -1e-8)
        @test abs(slope(ξ)) > 0.1        # the defect this replaces returned exactly 0.0
    end
    # and it is one curve through zero, not two pieces meeting there
    @test slope(1e-8) ≈ slope(-1e-8) rtol = 1e-4

    # The series is taken for |ξz| <= 0.01 and the direct form outside it, so the
    # density has to agree across that seam.
    z = (5.0 - 4.0) / 0.8
    inside, outside = 0.0099 / z, 0.0101 / z
    @test gev_logpdf(5.0, 4.0, 0.8, inside) ≈ gev_logpdf(5.0, 4.0, 0.8, outside) rtol = 1e-3
    @test gp_logpdf(3.0, 2.0, 0.0099 / 1.5) ≈ gp_logpdf(3.0, 2.0, 0.0101 / 1.5) rtol = 1e-3

    # Against Distributions.jl on both sides of the seam, where it is defined.
    for ξ in (0.05, 0.005, -0.005, -0.05)
        @test gev_logpdf(5.0, 4.0, 0.8, ξ) ≈
            logpdf(GeneralizedExtremeValue(4.0, 0.8, ξ), 5.0)
        @test gp_logpdf(3.0, 2.0, ξ) ≈ logpdf(GeneralizedPareto(0.0, 2.0, ξ), 3.0)
    end
end

@testset "design_matrix" begin
    @test design_matrix(3, nothing) == ones(3, 1)
    @test design_matrix(3, []) == ones(3, 1)
    @test design_matrix(3, [[1.0, 2.0, 3.0]]) == [1.0 1.0; 1.0 2.0; 1.0 3.0]
    @test size(design_matrix(4, [[1, 2, 3, 4], [5, 6, 7, 8]])) == (4, 3)
    @test_throws DimensionMismatch design_matrix(3, [[1.0, 2.0]])
end

@testset "gevfit, stationary" begin
    truth = GeneralizedExtremeValue(4.0, 0.8, 0.15)
    y = rand(MersenneTwister(543), truth, 5_000)
    fit = gevfit(y)
    @test_throws MethodError gevfit(y; prior_scale=1.0)   # maximum likelihood has no prior

    dists = getdistribution(fit)
    @test length(dists) == 1        # stationary, so one distribution covers every year
    d = only(dists)
    @test d.μ ≈ truth.μ rtol = 0.05
    @test d.σ ≈ truth.σ rtol = 0.05
    @test d.ξ ≈ truth.ξ atol = 0.04

    @test only(returnlevel(fit, 100)) ≈ quantile(truth, 0.99) rtol = 0.10
    @test length(params(fit).β_location) == 1
    @test isfinite(loglike(fit))
end

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

@testset "the shape does not get stuck at zero" begin
    # Both densities fall back to a ξ = 0 limit that does not contain ξ, so the
    # gradient there is identically zero and an optimizer started at exactly
    # zero reports success without ever moving the shape. Positive, negative and
    # near-zero truths all have to come back.
    for ξ in (0.15, -0.2, 0.0)
        truth = GeneralizedExtremeValue(4.0, 0.8, ξ)
        d = only(getdistribution(gevfit(rand(MersenneTwister(543), truth, 5_000))))
        @test d.ξ ≈ ξ atol = 0.04
    end
    for ξ in (0.2, -0.15)
        truth = GeneralizedPareto(0.0, 2.0, ξ)
        d = only(getdistribution(gpfit(rand(MersenneTwister(543), truth, 5_000), 0.0)))
        @test d.ξ ≈ ξ atol = 0.05
    end
end

@testset "gevfit, location trend" begin
    rng = MersenneTwister(543)
    n, slope = 4_000, 2.0
    t = collect(range(0, 1; length=n))
    y = [rand(rng, GeneralizedExtremeValue(4.0 + slope * t[i], 0.8, 0.1)) for i in 1:n]

    fit = gevfit(y; locationcov=[t])
    @test params(fit).β_location[2] ≈ slope rtol = 0.20   # the trend is recovered
    @test length(params(fit).β_logscale) == 1             # scale stayed stationary

    dists = getdistribution(fit)
    @test length(dists) == n                              # one per observation now
    @test dists[end].μ > dists[1].μ                       # and it moves the right way
    @test length(returnlevel(fit, 100)) == n
end

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

@testset "gevfit from a table" begin
    rng = MersenneTwister(543)
    n = 2_000
    df = DataFrame(; year=collect(range(0, 1; length=n)))
    df.lsl = [rand(rng, GeneralizedExtremeValue(4.0 + 2.0 * y, 0.8, 0.1)) for y in df.year]

    fit = gevfit(df, :lsl; locationcovid=[:year])
    @test params(fit).β_location[2] ≈ 2.0 rtol = 0.20
    @test length(getdistribution(fit)) == n

    # a NamedTuple of vectors is a table too, and takes no DataFrames to be one
    nt = (year=df.year, lsl=df.lsl)
    @test params(gevfit(nt, :lsl; locationcovid=[:year])).β_location[2] ≈ 2.0 rtol = 0.20
end

@testset "gpfit and the Bayesian fitters from a table" begin
    rng = MersenneTwister(543)
    df = DataFrame(; x=collect(range(0, 1; length=1_000)))
    df.q = rand(rng, GeneralizedPareto(10.0, 2.0, 0.1), 1_000)
    fit = gpfit(df, :q, 10.0; logscalecovid=[:x])
    @test length(params(fit).β_logscale) == 2
    bayes = gpfitbayes(df, :q, 10.0; n_samples=50, n_chains=1, rng=rng)
    @test length(posterior_distributions(bayes)) == 50
end

@testset "gpfit" begin
    threshold = 10.0
    truth = GeneralizedPareto(threshold, 2.0, 0.2)
    fit = gpfit(rand(MersenneTwister(543), truth, 5_000), threshold)

    d = only(getdistribution(fit))
    @test d.μ == threshold          # the level comes back, not the exceedance
    @test d.σ ≈ truth.σ rtol = 0.08
    @test d.ξ ≈ truth.ξ atol = 0.05
    # a POT fit's return period is in years only once it knows the rate
    @test only(returnlevel(fit, 50; rate=5.0)) ≈ pot_return_level(50, threshold, d.σ, d.ξ, 5.0)
    @test only(returnlevel(fit, 50; rate=5.0)) == returnlevel(d, 50; rate=5.0)
    @test_throws ArgumentError returnlevel(fit, 50)

    @test_throws ArgumentError gpfit([1.0, 2.0], 99.0)
end

@testset "gpfit keeps only the exceedances" begin
    rng = MersenneTwister(543)
    threshold = 10.0
    above = rand(rng, GeneralizedPareto(threshold, 2.0, 0.2), 3_000)
    fit = gpfit(vcat(above, fill(5.0, 500)), threshold)   # 500 non-exceedances
    @test length(fit.y) == 3_000
    @test only(getdistribution(fit)).σ ≈ 2.0 rtol = 0.10
end

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

@testset "gevfitbayes" begin
    truth = GeneralizedExtremeValue(4.0, 0.8, 0.15)
    y = rand(MersenneTwister(543), truth, 2_000)
    fit = gevfitbayes(y; n_samples=400, n_chains=2, rng=MersenneTwister(543))

    # Turing returns a FlexiChain, indexed by variable name. Each entry is the
    # whole coefficient vector for that draw, so `first` picks the intercept.
    chain = fit.estimate
    intercept(name) = mean(first.(chain[name]))
    @test intercept(:β_location) ≈ truth.μ rtol = 0.05
    @test exp(intercept(:β_logscale)) ≈ truth.σ rtol = 0.10
    @test intercept(:β_shape) ≈ truth.ξ atol = 0.06

    # a posterior is not a point estimate, and says so rather than guessing
    @test_throws ArgumentError params(fit)
    @test_throws ArgumentError loglike(fit)

    # one stationary distribution per draw, matching the chain
    draws = posterior_distributions(fit)
    @test length(draws) == 400 * 2
    @test all(d -> length(d) == 1, draws)
    @test median(only(d).σ for d in draws) ≈ truth.σ rtol = 0.10
    @test returnlevel(only(first(draws)), 100) ≈ quantile(only(first(draws)), 0.99)

    # a point estimate has no draws
    @test_throws ArgumentError posterior_distributions(gevfit(y))
end

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

@testset "gpfitbayes and posterior_distributions" begin
    threshold = 10.0
    truth = GeneralizedPareto(threshold, 2.0, 0.2)
    y = rand(MersenneTwister(543), truth, 1_000)
    fit = gpfitbayes(y, threshold; n_samples=300, n_chains=2, rng=MersenneTwister(543))

    draws = posterior_distributions(fit)
    @test length(draws) == 300 * 2
    @test all(d -> only(d).μ == threshold, draws)
    @test median(only(d).σ for d in draws) ≈ truth.σ rtol = 0.10
    levels = [returnlevel(only(d), 100; rate=3.0) for d in draws]
    @test all(>(threshold), levels)
end

@testset "detrend_baseline and recentre" begin
    yrs = collect(2000:2019)
    msl = 0.01 .* (yrs .- 2000)                    # 10 mm/yr of sea-level rise
    annmax = 5.0 .+ msl                            # maxima that ride on it exactly

    @test detrend_baseline(yrs, annmax, msl, :none) == zeros(20)

    for method in (:linear, :msl)
        baseline = detrend_baseline(yrs, annmax, msl, method)
        flat = annmax .- baseline .+ recentre(baseline)
        @test all(≈(flat[1]), flat)                # the rise is gone either way
        @test mean(flat) ≈ mean(annmax[(end - 4):end]) rtol = 1e-8   # recentred on the last 5
    end

    # Only :msl follows a baseline that is not a straight line. Give the sea a
    # step change and the fitted line smears it across the record.
    stepped = [y < 2010 ? 0.0 : 0.5 for y in yrs]
    surge = fill(5.0, 20) .+ stepped
    by_msl = surge .- detrend_baseline(yrs, surge, stepped, :msl)
    by_line = surge .- detrend_baseline(yrs, surge, stepped, :linear)
    @test std(by_msl) ≈ 0 atol = 1e-10             # removed exactly
    @test std(by_line) > 0.1                       # a line cannot see a step

    @test_throws ArgumentError detrend_baseline(yrs, annmax, msl, :quadratic)

    # one point defines no line, and a NaN baseline would be silent
    @test_throws ArgumentError detrend_baseline([2000], [5.0], [0.0], :linear)
    @test detrend_baseline([2000], [5.0], [0.0], :msl) == [0.0]

    # recentre averages the last REF_WINDOW entries, and copes with a record
    # shorter than that window.
    @test recentre([1.0, 2.0, 3.0]) ≈ 2.0
    @test recentre(collect(1.0:10.0)) ≈ mean(6.0:10.0)
end

@testset "records carry their station and units" begin
    station = Station("8638610"; name="Sewells Point", datum="MSL")
    @test occursin("Sewells Point", sprint(show, station))
    @test occursin("MSL", sprint(show, station))

    times = [DateTime(2020, 1, 1, h) for h in 0:3]
    record = WaterLevelRecord(station, times, [1.0, 1.1, 1.2, 1.3], u"m")
    @test length(record) == 4
    @test record[1].level == 1.0u"m"
    @test record[1].time == DateTime(2020, 1, 1)
    @test last(collect(record)).level == 1.3u"m"      # iteration gives readings
    @test waterlevels(record) == [1.0, 1.1, 1.2, 1.3]u"m"
    @test obstimes(record) == times
    @test occursin("4 readings", sprint(show, MIME"text/plain"(), record))

    # The unit is one field for the whole record, so the levels stay bare floats.
    @test record.levels isa Vector{Float64}
    @test ustrip.(waterlevels(record)) == record.levels

    @test_throws DimensionMismatch WaterLevelRecord(station, times, [1.0], u"m")
    @test_throws ArgumentError WaterLevelRecord(station, times, ones(4), u"s")
end

@testset "AnnMaxRecord takes the maximum of each year" begin
    station = Station("test")
    # Two years of six-hourly readings, the second 0.3 m higher throughout.
    times, levels = DateTime[], Float64[]
    for (i, yr) in enumerate((2000, 2001)), day in 1:365, hour in (0, 6, 12, 18)
        push!(times, DateTime(yr, 1, 1) + Day(day - 1) + Hour(hour))
        push!(levels, (i - 1) * 0.3 + sin(day / 58) * 0.5 + hour / 100)
    end
    record = WaterLevelRecord(station, times, levels, u"m")

    annmax = AnnMaxRecord(record; detrend=:none, min_readings=1_000)
    @test length(annmax) == 2
    @test obsyears(annmax) == [2000, 2001]
    @test unit(first(waterlevels(annmax))) == u"ft"    # converted on the way out
    @test annmax.detrend === :none
    @test all(iszero, annmax.baseline)
    # 2001 sits 0.3 m above 2000, and :none leaves that difference alone.
    @test ustrip(u"m", waterlevels(annmax)[2] - waterlevels(annmax)[1]) ≈ 0.3 atol = 1e-8

    # Units convert rather than relabel: the same record in metres and in feet.
    in_metres = AnnMaxRecord(record; detrend=:none, min_readings=1_000, units=u"m")
    @test ustrip.(u"m", waterlevels(annmax)) ≈ in_metres.levels

    # A year with too few readings is dropped rather than half-counted.
    @test_throws ArgumentError AnnMaxRecord(record; min_readings=10_000)
    @test_throws ArgumentError AnnMaxRecord(record; detrend=:quadratic)
end

@testset "detrend removes the trend from every reading" begin
    station = Station("test")
    # Four years of six-hourly readings on a sea rising 0.1 m per year.
    times, levels = DateTime[], Float64[]
    for (i, yr) in enumerate(2000:2003), day in 1:365, hour in (0, 6, 12, 18)
        push!(times, DateTime(yr, 1, 1) + Day(day - 1) + Hour(hour))
        push!(levels, 0.1 * (i - 1) + sin(day / 58) * 0.5)
    end
    record = WaterLevelRecord(station, times, levels, u"ft")

    for method in (:msl, :linear)
        flat = detrend(record; method=method, min_readings=1_000)
        @test unit(first(waterlevels(flat))) == u"ft"      # the unit is kept
        @test length(flat) == length(record)
        yearly = [mean(flat.levels[year.(obstimes(flat)) .== yr]) for yr in 2000:2003]
        @test all(≈(yearly[end]), yearly)                   # every year now sits level
        # re-centred on the last five years, which here is all four of them
        @test yearly[end] ≈ mean(levels) atol = 1e-8
    end

    unchanged = detrend(record; method=:none, min_readings=1_000)
    @test unchanged.levels ≈ record.levels

    # the annual maxima of the detrended readings carry no trend either
    annmax = AnnMaxRecord(detrend(record; min_readings=1_000); detrend=:none, min_readings=1_000)
    @test all(≈(annmax.levels[1]), annmax.levels)

    # a sparse year is dropped, so the kept years count the record length
    partial = WaterLevelRecord(
        station, vcat(times, [DateTime(2004, 1, 1)]), vcat(levels, [0.0]), u"ft"
    )
    @test unique(year.(obstimes(detrend(partial; min_readings=1_000)))) == 2000:2003

    @test_throws ArgumentError detrend(record; method=:quadratic)
    @test_throws ArgumentError detrend(record; min_readings=10_000)
end

@testset "DataFrame(record)" begin
    station = Station("test")
    record = WaterLevelRecord(
        station, [DateTime(2020, 1, 1), DateTime(2020, 1, 1, 1)], [1.0, 2.0], u"m"
    )

    df = DataFrame(record)
    @test names(df) == ["time", "level_m"]
    @test df.level_m == [1.0, 2.0]                     # bare numbers by default
    @test DataFrame(record; keep_units=true).level_m == [1.0, 2.0]u"m"

    annmax = AnnMaxRecord(station, [2000, 2001], [4.0, 5.0], u"ft", :linear, [0.1, 0.2])
    adf = DataFrame(annmax)
    @test names(adf) == ["year", "level_ft", "baseline_ft"]
    @test adf.year == [2000, 2001]
    @test DataFrame(annmax; keep_units=true).level_ft == [4.0, 5.0]u"ft"

    @test_throws ArgumentError AnnMaxRecord(
        station, [2000], [4.0], u"ft", :quadratic, [0.0]
    )
end

@testset "load_water_level rejects an unusable cache" begin
    path = joinpath(mktempdir(), "wrong-columns.csv")
    CSV.write(path, DataFrame(; time=[DateTime(2000)], surge=[4.0]))   # no level_m
    @test_throws ArgumentError load_water_level("8638610"; cache=path)
end

@testset "load_water_level reads a cache without the network" begin
    path = joinpath(mktempdir(), "cached.csv")
    CSV.write(
        path,
        DataFrame(;
            time=[DateTime(2000, 1, 1), DateTime(2000, 1, 1, 1)], level_m=[1.0, 2.0]
        ),
    )

    record = load_water_level("not-a-station"; cache=path)   # a download would fail
    @test length(record) == 2
    @test waterlevels(record) == [1.0, 2.0]u"m"
    @test record.station.id == "not-a-station"
end

@testset "default cache name carries datum and years" begin
    @test CEVE543Utils.default_cache("8638610", "NAVD88", 1990, 2000) ==
        "8638610-NAVD88-1990-2000-hourly.csv"
end

# The NOAA API is a network dependency and a full record is a large download, so
# these run only when asked for:
#     CEVE543_NETWORK_TESTS=1 julia --project=. test/runtests.jl
if get(ENV, "CEVE543_NETWORK_TESTS", "0") == "1"
    @testset "load_water_level against NOAA" begin
        path = joinpath(mktempdir(), "sewells.csv")
        record = load_water_level("8638610"; first_year=2003, last_year=2004, cache=path)

        @test length(record) == 17_544                  # two years of hourly readings
        @test unit(first(waterlevels(record))) == u"m"
        # Hurricane Isabel, the largest reading in this window.
        @test maximum(record.levels) ≈ 1.992 atol = 1e-3
        @test isfile(path)                              # the download was cached
        @test length(load_water_level("8638610"; cache=path)) == length(record)

        annmax = AnnMaxRecord(record; detrend=:none, units=u"m")
        @test obsyears(annmax) == [2003, 2004]
        @test maximum(annmax.levels) ≈ 1.992 atol = 1e-3

        @test_throws ArgumentError load_water_level(
            "0000000"; first_year=2020, last_year=2020, cache=nothing
        )
    end
end

@testset "decluster" begin
    times = [DateTime(2000, 1, 1, h) for h in 0:23]
    levels = Float64[h == 5 ? 10.0 : h == 6 ? 9.0 : h == 20 ? 12.0 : 1.0 for h in 0:23]
    idx, exc = decluster(times, levels, 5.0; min_gap_hours=6)
    @test length(idx) == 2
    @test levels[idx[1]] == 10.0
    @test levels[idx[2]] == 12.0
    @test exc == [5.0, 7.0]

    idx0, exc0 = decluster(times, levels, 99.0)
    @test isempty(idx0) && isempty(exc0)

    @test_throws ArgumentError decluster(reverse(times), levels, 5.0)
    # a gap exactly equal to the minimum does not start a new cluster
    two = [DateTime(2000, 1, 1, 0), DateTime(2000, 1, 1, 6)]
    @test length(decluster(two, [10.0, 9.0], 5.0; min_gap_hours=6)[1]) == 1
end

@testset "decluster_daily" begin
    vals = zeros(20)
    vals[3] = 10.0; vals[4] = 11.0; vals[15] = 8.0
    idx, exc = decluster_daily(vals, 5.0; min_gap_days=3)
    @test length(idx) == 2
    @test vals[idx[1]] == 11.0
    @test exc[2] ≈ 3.0

    @test isempty(decluster_daily(vals, 99.0)[1])
end

@testset "mrl_data" begin
    rng = MersenneTwister(543)
    σ = 2.0
    data = rand(rng, Exponential(σ), 10_000)
    t, m, lo, hi = mrl_data(data; n_thresholds=50)
    @test length(t) == 50
    ok = .!isnan.(m)
    @test any(ok)
    @test all(hi[ok] .- lo[ok] .> 0)
    @test hi[ok][1] - lo[ok][1] < hi[ok][end] - lo[ok][end]   # fewer exceedances, wider band
    # Memoryless: mean excess above any threshold is σ, regardless of threshold.
    @test mean(m[ok]) ≈ σ rtol = 0.15

    # an explicit grid and count floor
    t2, m2, _, _ = mrl_data(data; n_thresholds=5, lo=1.0, hi=9.0, min_count=50)
    @test t2 == collect(range(1.0, 9.0; length=5))
    @test isfinite(m2[1])
    @test isnan(mrl_data(data; n_thresholds=3, lo=1.0, hi=40.0, min_count=50)[2][end])
end

@testset "stability_data" begin
    rng = MersenneTwister(543)
    data = rand(rng, GeneralizedPareto(0.0, 2.0, 0.1), 10_000)
    t, ss, xi, ss_se, xi_se = stability_data(data; n_thresholds=20)
    ok = .!isnan.(xi)
    @test any(ok)
    @test mean(xi[ok]) ≈ 0.1 atol = 0.15
    # σ* = σ_u - ξu is constant in u when the data are GPD above every threshold,
    # so every estimate sits within a few standard errors of the truth. The
    # defect this replaces (σ/√n) put the upper thresholds thirty errors out.
    @test all(abs.(ss[ok] .- 2.0) .<= 3 .* ss_se[ok])
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

@testset "pot_return_level" begin
    sigma, xi, u, lam = 2.0, 0.2, 5.0, 10.0
    rl = pot_return_level(100, u, sigma, xi, lam)
    @test rl > u
    d = GeneralizedPareto(0.0, sigma, xi)
    expected = u + quantile(d, 1 - 1 / (100 * lam))
    @test rl ≈ expected rtol = 0.01

    rl_gumbel = pot_return_level(100, u, sigma, 0.0, lam)
    @test rl_gumbel ≈ u + sigma * log(100 * lam)
end

@testset "return_period_axis, from the Makie extension" begin
    ticks = [1, 2, 5, 10, 25, 100, 1000]

    ax = Axis(Figure()[1, 1]; ylabel="level (ft)", title="kept")
    scatter!(ax, [1.0, 10.0, 100.0], [1.0, 2.0, 3.0])
    return_period_axis!(ax)
    @test ax.xscale[] === log10
    @test ax.xticks[] == (ticks, string.(ticks))
    @test ax.ylabel[] == "level (ft)"        # left alone
    @test ax.title[] == "kept"               # left alone
    @test ax.finallimits[].origin[1] < 1.0   # autoscaled to the data, not pinned

    # an empty axis still has limits starting at zero, which log10 rejects
    empty_ax = return_period_axis!(Axis(Figure()[1, 1]))
    @test empty_ax.xscale[] === log10
    @test empty_ax.finallimits[].origin[1] ≈ 1.0   # pinned to the tick span instead

    built = return_period_axis(Figure()[1, 1]; ylabel="level (ft)")
    @test built.xscale[] === log10
    @test built.ylabel[] == "level (ft)"
    @test first(built.xticks[]) == ticks
end

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
