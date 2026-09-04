import Pkg;
Pkg.activate(@__DIR__);

begin
    using Pipe, ProgressMeter
    using LinearAlgebra, Distances, Distributions
    using Random, Statistics, VectorizedStatistics
    using Plots, Plots.Measures, LaTeXStrings
    using JLD2
end

for filename in readdir("src")
    if endswith(filename, ".jl")
        includet(joinpath("src", filename))
    end
end


############
# Three extra noise models, beyond additive / additiveAbs / multiplicative in src/utils.jl:
#   rounding   d = floor(δ) + B,   B | Δ ~ Bernoulli(frac(δ))
#   missing    d = (M/π(δ)) * δ,         π(δ) = π0 + (1-π0)exp(-δ),  M | Δ ~ Bernoulli(π(δ))
#   sparse     d = δ + c*δ*S,            S = 1{δ far in row i and row j}, deterministic
# rounding/missing use a Uniform[-1/2, 1/2] driver Ξ instead of the t_q driver above;
# sparse ignores Ξ entirely.

fracpart(z) = z - floor(z)

function far_mask(Δ, α)
    n = size(Δ, 1)
    k = ceil(Int, α * (n - 1))
    k <= 0 && return falses(n, n)
    S = trues(n, n)
    if k < n - 1
        thr = [partialsort(Δ[i, :], k + 1, rev=true) for i in 1:n]
        row = Δ .> thr
        S = row .& permutedims(row)
    end
    S[diagind(S)] .= false
    return S
end

function noise_driver(n, noise; q=5, sigma=0.1)
    Ξ = noise in (:rounding, :missing, :sparse) ?
        Matrix(Symmetric(rand(Uniform(-0.5, 0.5), n, n))) :
        Matrix(Symmetric(rand(TDist(q), n, n))) .* sigma
    Ξ[diagind(Ξ)] .= 0.0
    return Ξ
end

function Dist(Δ, Ξ; sigma=0.1, noise=:additive, u=0.25, pi0=0.6, a=1.0, decay=true, c=1.0)
    if noise == :additive
        D = Δ .+ (Ξ .* sigma)
    elseif noise == :additiveAbs
        σΞ = sigma .* Ξ
        D = Δ .+ (σΞ .^ 2) .+ (2 .* σΞ .* sqrt.(Δ))
    elseif noise == :multiplicative
        D = Δ .* (1 .+ (Ξ .* sigma))
    elseif noise == :rounding
        F = fracpart.(Δ ./ u)
        B = (Ξ .+ 0.5) .<= F
        D = Δ .+ u .* (B .- F)
    elseif noise == :missing
        Π = pi0 .+ (1 - pi0) .* exp.(-Δ)
        M = (Ξ .+ 0.5) .<= Π
        D = Δ .+ (Δ ./ Π) .* (M .- Π)
    elseif noise == :sparse
        n = size(Δ, 1)
        S = far_mask(Δ, decay ? a / sqrt(n) : a)
        D = Δ .+ c .* Δ .* S
    end
    D[diagind(D)] .= 0.0
    return D
end

# scale the error should track: rms conditional sd of ε for the unbiased models;
# for :sparse the noise is deterministic given Δ, so it's the (A4) budget instead
function sigma_eff(Δ, Ξ; sigma=0.1, noise=:additive, u=0.25, pi0=0.6, a=1.0, decay=true, c=1.0)
    n = size(Δ, 1)
    iu = triu(trues(n, n), 1)
    if noise == :rounding
        F = fracpart.(Δ ./ u)
        V = (u^2) .* F .* (1 .- F)
        return sqrt(mean(V[iu]))
    elseif noise == :missing
        Π = pi0 .+ (1 - pi0) .* exp.(-Δ)
        V = (Δ .^ 2) .* (1 .- Π) ./ Π
        return sqrt(mean(V[iu]))
    elseif noise == :sparse
        M = c .* Δ .* far_mask(Δ, decay ? a / sqrt(n) : a)
        μ = mean(M, dims=2)
        HMH = M .- μ .- μ' .+ mean(μ)
        return maximum(sum(abs, HMH, dims=2)) / sqrt(n)
    else
        E = Dist(Δ, Ξ; sigma=sigma, noise=noise) .- Δ
        return std(E[iu])
    end
end


function simulation(n; d=3, q=5, sigma=0.1, kappa=1.0, R=1.0, seed=0, noise=:additive, level=NamedTuple())
    if seed != 0
        Random.seed!(2025 + seed)
    end
    Xn = randBall(n, d=d)
    Xn .= Xn |>
            x -> map(
                x -> norm(x) < R ? x : x .* (R / norm(x)),
                eachrow(x)
        ) |>
        x -> hcat(x...) |> permutedims
    Σ = diagm(0 => range(1 / kappa, kappa, length=d))
    Xn .= Xn * Σ
    Δ = pairwise(SqEuclidean(), Xn, dims=1)
    Ξ = noise_driver(n, noise; q=q, sigma=sigma)
    D = Dist(Δ, Ξ; sigma=sigma, noise=noise, level...)
    σ = sigma_eff(Δ, Ξ; sigma=sigma, noise=noise, level...)
    Xnhat = mds(D, d) |> (x -> procrustes(x, Xn))
    Error = norm.(eachrow(Xnhat - Xn))
    maximum(Error), sqrt(mean(Error .^ 2)), σ
end



NoiseLevels = Dict(
    :rounding => [(u=1.0,)],
    :missing => [(pi0=0.9,), (pi0=0.6,), (pi0=0.3,)],
    :sparse => [(a=0.5, decay=true), (a=1.0, decay=true), (a=0.05, decay=false)],
)
NoiseLabels = Dict(
    :rounding => [""],
    :missing => [L"\pi_0 = 0.9", L"\pi_0 = 0.6", L"\pi_0 = 0.3"],
    :sparse => [L"\alpha_n = 0.5 × n^{-1/2}", L"\alpha_n = n^{-1/2}", L"\alpha_n \equiv 0.05"],
)
NoiseTitles = Dict(
    :rounding => "Random rounding",
    :missing => "Missing at random",
    :sparse => "Sparse noise",
)

Ns = [250; 500; 1000; 2500; 5000]
Ks = [1.0]
Noises = [:rounding, :missing, :sparse]
reps = 20

sim_res = Dict(
    noise => @showprogress [
        simulation(n; d=3, kappa=k, seed=r, noise=noise, level=lvl)
        for (n, k, lvl, r) in Iterators.product(Ns, Ks, NoiseLevels[noise], 1:reps)
    ]
    for noise in Noises
)

# jldsave(
#     "results/additional-simulations.jld2";
#     sim_res=sim_res, Ns=Ns, Ks=Ks, Noises=Noises, Levels=NoiseLevels
# )




############
# Make plots
begin
    agg(x, f, dims=4) = dropdims(f(x, dims=dims), dims=dims)
    quant(alpha) = (x; kwargs...) -> vquantile(x, alpha; kwargs...)
    cls3 = [:chartreuse2 :dodgerblue1 :firebrick1]
    has_rate(noise, s) = !(noise == :sparse && s == 3)
end

function loss_panel(noise, row)
    component = row == 1 ? 2 : 1   # x[2] = L_rmse, x[1] = L_2->inf
    res = map(x -> x[component], sim_res[noise])
    m = agg(res, median)[:, 1, :]
    hi = agg(res, quant(0.9))[:, 1, :]
    lo = agg(res, quant(0.1))[:, 1, :]
    p = plot(Ns,
        m,
        ribbon=(m .- lo, hi .- m),
        fa=0.15, m=:o, ms=3, msw=0, lw=1.5,
        c=cls3,
        ls=noise == :sparse ? [:solid :solid :dash] : [:solid :solid :solid],
        label=permutedims(NoiseLabels[noise]),
        legend=:bottomleft,
        legendfontsize=8,
        foreground_color_legend=nothing,
        title=row == 1 ? NoiseTitles[noise] : "",
        titlefontsize=11,
        xlabel=row == 2 ? L"n" : "",
        ylabel=noise == Noises[1] ? (row == 1 ? L"L${}_{\mathrm{rmse}}$" : L"L${}_{2\!\!\to\!\!\infty}$") : "",
        grid=false,
    )
    for s in 1:size(m, 2)
        has_rate(noise, s) || continue
        rate = row == 1 ?
            n -> m[1, s] * sqrt(Ns[1] / n) :
            n -> m[1, s] * sqrt((log(n) / n) / (log(Ns[1]) / Ns[1]))
        p = plot!(p, Ns, rate, c=:gray30, ls=:dash, lw=0.8, label="")
    end
    p
end

begin
    plts = [loss_panel(noise, row) for noise in Noises, row in 1:2]
    p3 = plot(plts...,
        layout=(2, 3), size=(1050, 640), scale=:log10,
        left_margin=5 * Plots.mm, bottom_margin=4 * Plots.mm, top_margin=2 * Plots.mm,
    )
end
savefig(p3, "plots/p3-additional-noise.pdf")
