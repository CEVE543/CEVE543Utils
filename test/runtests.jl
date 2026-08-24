using CairoMakie          # loading a backend is what turns the Makie extension on
using CEVE543Utils
using DataFrames
using Distributions
using Random
using Statistics
using Test

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
    fit = gevfit(rand(MersenneTwister(543), truth, 5_000))

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

@testset "gpfit" begin
    threshold = 10.0
    truth = GeneralizedPareto(threshold, 2.0, 0.2)
    fit = gpfit(rand(MersenneTwister(543), truth, 5_000), threshold)

    d = only(getdistribution(fit))
    @test d.μ == threshold          # the level comes back, not the exceedance
    @test d.σ ≈ truth.σ rtol = 0.08
    @test d.ξ ≈ truth.ξ atol = 0.05
    @test only(returnlevel(fit, 50)) > threshold

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
