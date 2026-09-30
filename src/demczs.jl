# Differential evolution MCMC with an archive of past states and snooker updates,
# as an AbstractMCMC sampler that Turing runs through `externalsampler`.
# Written here because DifferentialEvolutionMCMC.jl takes its own model type rather
# than a Turing model, and its defaults are not DE-MCzs.

"""
    DEMCzs(; n_walkers=3, n_burnin=5_000, thin=10, snooker=0.1, thin_archive=10)

DE-MCzs (ter Braak & Vrugt 2008). Run it on a Turing model with
`sample(model, externalsampler(DEMCzs()), MCMCThreads(), n_samples, n_chains)`.

Each chain is `n_walkers` walkers in unconstrained space sharing an archive `Z`.
A sweep updates every walker once; a step runs `thin` sweeps and returns the first
walker's state. From `x`, with `z, z₁, z₂` distinct draws from `Z` and `d = length(x)`:

  - with probability `snooker`, `x′ = x + γ (P z₁ - P z₂)`, `P` the projection onto
    `x - z`, `γ ~ U(1.2, 2.2)`, accepted with ratio `π(x′)/π(x) (‖x′ - z‖/‖x - z‖)^(d-1)`;
  - otherwise `x′ = x + γ (z₁ - z₂) + e`, `γ = 2.38/√(2d)` (`γ = 1` with probability
    0.1), `e ~ N(0, 10⁻¹² I)`, accepted with ratio `π(x′)/π(x)`.

`Z` starts as Turing's initial point and `max(10d, n_walkers + 3) - 1` draws from
`U(-2, 2)ᵈ` with finite density; the walkers start at its first `n_walkers`
entries. Every `thin_archive` sweeps the walkers are appended to `Z`. The first
`n_burnin` sweeps run before the first draw is returned.
"""
struct DEMCzs <: AbstractMCMC.AbstractSampler
    n_walkers::Int
    n_burnin::Int
    thin::Int
    snooker::Float64
    thin_archive::Int
end

DEMCzs(; n_walkers=3, n_burnin=5_000, thin=10, snooker=0.1, thin_archive=10) =
    DEMCzs(n_walkers, n_burnin, thin, snooker, thin_archive)

mutable struct DEMCzsState{T<:Real}
    walkers::Vector{Vector{T}}
    logp::Vector{T}
    archive::Vector{Vector{T}}
    sweeps::Int
    accepted::Bool   # whether the first walker moved on the last sweep
end

AbstractMCMC.getparams(s::DEMCzsState) = first(s.walkers)
AbstractMCMC.getstats(s::DEMCzsState) = (accepted=s.accepted,)

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::DEMCzs;
    initial_params,
    kwargs...,
)
    logp(x) = LogDensityProblems.logdensity(model.logdensity, x)
    x₀ = collect(initial_params)
    d = length(x₀)
    n_archive = max(10d, spl.n_walkers + 3)
    archive = [x₀]
    for _ in 1:(1_000 * n_archive)
        length(archive) == n_archive && break
        x = 4 .* rand(rng, eltype(x₀), d) .- 2
        isfinite(logp(x)) && push!(archive, x)
    end
    length(archive) == n_archive ||
        error("DEMCzs found $(length(archive)) of $n_archive starting points with finite density")
    walkers = copy.(archive[1:spl.n_walkers])
    s = DEMCzsState(walkers, logp.(walkers), archive, 0, true)
    for _ in 1:spl.n_burnin
        sweep!(rng, logp, spl, s)
    end
    return AbstractMCMC.getparams(s), s
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::DEMCzs,
    s::DEMCzsState;
    kwargs...,
)
    logp(x) = LogDensityProblems.logdensity(model.logdensity, x)
    for _ in 1:spl.thin
        sweep!(rng, logp, spl, s)
    end
    return AbstractMCMC.getparams(s), s
end

function sweep!(rng, logp, spl::DEMCzs, s::DEMCzsState)
    for j in eachindex(s.walkers)
        moved = update!(rng, logp, spl, s, j)
        j == 1 && (s.accepted = moved)
    end
    s.sweeps += 1
    s.sweeps % spl.thin_archive == 0 && append!(s.archive, copy.(s.walkers))
    return s
end

# One Metropolis-Hastings update of walker `j`; returns whether it moved.
function update!(rng, logp, spl::DEMCzs, s::DEMCzsState, j)
    x = s.walkers[j]
    d = length(x)
    z, z₁, z₂ = s.archive[distinct3(rng, length(s.archive))]
    if rand(rng) < spl.snooker
        u = x - z
        uu = dot(u, u)
        iszero(uu) && return false   # a walker equal to its archived copy has no snooker direction
        x′ = x + rand(rng, Uniform(1.2, 2.2)) * (dot(z₁ - z₂, u) / uu) * u
        log_jacobian = (d - 1) * (log(norm(x′ - z)) - log(sqrt(uu)))
    else
        γ = rand(rng) < 0.1 ? 1.0 : 2.38 / sqrt(2d)
        x′ = x + γ * (z₁ - z₂) + 1e-6 * randn(rng, d)
        log_jacobian = 0.0
    end
    lp′ = logp(x′)
    log(rand(rng)) < lp′ - s.logp[j] + log_jacobian || return false   # false for a NaN density
    s.walkers[j] = x′
    s.logp[j] = lp′
    return true
end

function distinct3(rng, n)
    i = rand(rng, 1:n)
    j = rand(rng, 1:n)
    while j == i
        j = rand(rng, 1:n)
    end
    k = rand(rng, 1:n)
    while k == i || k == j
        k = rand(rng, 1:n)
    end
    return [i, j, k]
end
