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


function simulation(n; d=3, q=5, sigma=0.1, kappa=1.0, R=1.0, seed=0, noise=:additive)
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
    Ξ = Symmetric(rand(TDist(q), n, n)) .* sigma
    Ξ[diagind(Ξ)] .= 0
    D = Dist(Δ, Ξ; noise=noise)
    Xnhat = mds(D, d) |> (x -> procrustes(x, Xn))
    Error = norm.(eachrow(Xnhat - Xn))
    maximum(Error), sqrt(mean(Error .^ 2))
end



Ns = [250; 500; 1000; 2500; 5000]
Qs = [2.51, 5, 7]
Sigmas = [0.1, 0.25, 0.5]
Ks = range(1.0, 2.0, length=5)
Noises = [:additive, :additiveAbs, :multiplicative]
reps = 5

sim_res = Dict(
    noise => @showprogress [
        simulation(n; d=3, q=q, sigma=s, kappa=k, seed=r, noise=noise)
        # for n in Ns, q in Qs, k in Ks, s in Sigmas, r in 1:5
        for (n, q, k, s, r) in Iterators.product(Ns, Qs, Ks, Sigmas, 1:reps)
    ]
    for noise in Noises
)

jldsave(
    "results/simulations.jld2"; 
    sim_res=sim_res, Ns=Ns, Qs=Qs, 
    Sigmas=Sigmas, Ks=Ks, Noises=Noises
)