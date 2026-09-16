import Pkg;
Pkg.activate(@__DIR__);

begin
    using Pipe, ProgressMeter
    using LinearAlgebra, Distances, Distributions
    using Random, Statistics, VectorizedStatistics
    using JLD2
end

for filename in readdir("src")
    if endswith(filename, ".jl")
        includet(joinpath("src", filename))
    end
end

function simulation(rng, n; d=3, q=5, sigma=0.1, kappa=1.0, R=1.0, noise=:additive)
    Xn = randBall(rng, n, d=d)
    Xn .= Xn |>
            x -> map(
                x -> norm(x) < R ? x : x .* (R / norm(x)),
                eachrow(x)
        ) |>
        x -> hcat(x...) |> permutedims
    Σ = diagm(0 => range(1 / kappa, kappa, length=d))
    Xn .= Xn * Σ
    Δ = pairwise(SqEuclidean(), Xn, dims=1)
    Ξ = Symmetric(rand(rng, TDist(q), n, n)) .* sigma
    Ξ[diagind(Ξ)] .= 0
    D = Dist(Δ, Ξ; noise=noise)
    Xnhat = mds(D, d) |> (x -> procrustes(x, Xn))
    Error = norm.(eachrow(Xnhat - Xn))
    return sqrt(mean(Error .^ 2)), maximum(Error)
end

# Simulation parameters

rng = Xoshiro(2026)

Ns = [250; 500; 1000; 2500; 5000]
Qs = [2.5, 4.5, 8.5]
Sigmas = [0.1, 0.2, 0.4]
Ks = range(1.0, 3.0, length=5)
Noises = [:additive, :additiveAbs, :multiplicative]
reps = 30

sim_res = Dict(
    noise => @showprogress [
        simulation(rng, n; d=3, q=q, sigma=s, kappa=k, noise=noise)
        for (n, q, k, s, r) in Iterators.product(Ns, Qs, Ks, Sigmas, 1:reps)
    ]
    for noise in Noises
)

jldsave(
    "results/simulations-finale.jld2"; 
    sim_res=sim_res, Ns=Ns, Qs=Qs, 
    Sigmas=Sigmas, Ks=Ks, Noises=Noises
)