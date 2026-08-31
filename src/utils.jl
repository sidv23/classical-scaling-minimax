using LinearAlgebra, Arpack, Random
using Distances, Distributions, Statistics
using Plots

tup(X) = Tuple.(eachrow(X))
center!(X) = X .= X .- mean(X, dims=1)
center(X) = X .- mean(X, dims=1)

function m2t(x)
    return Tuple.(eachrow(x))
end

function m2v(x)
    return [[v...] for v in tup(x)]
end

#### GR params
gr_params = (;
    format=:png,
    size=(350, 350),
    msw=0.0,
    ma=0.75,
    leg=:outertop,
    legendfontsize=10,
    legend_column=-1,
    fg_color_legend=nothing,
    left_margin=-3 * Plots.mm,
    bottom_margin=-3 * Plots.mm,
    top_margin=-3 * Plots.mm,
);



function doubleCentering(D)
    μ = mean(D, dims=2)
    return Symmetric(-0.5 .* (D .- μ .- μ' .+ mean(μ)))
end

function slowmds(D::Matrix, d::Int)
    E = doubleCentering(D)
    λ, V = eigen(E)
    λ_top, V_top = λ[end-d+1:end], V[:, end-d+1:end]
    L = V_top * diagm(.√max.(0, λ_top))
    return L
end

function mds(D::Matrix, d::Int)
    E = doubleCentering(D)
    # λ, V = eigen(E)
    # L = V[:, end:-1:end-d+1] .* .√λ[end:-1:end-d+1]'
    λ, V = eigs(E, nev=d, which=:LR)
    L = V .* .√λ'
    return L
end

function Dist(Δ, Ξ; sigma=0.1, noise=:additive)
    if noise == :additive
        D = Δ .+ (Ξ .* sigma)
        # p = 1
    elseif noise == :additiveAbs
        σΞ = sigma .* Ξ
        D = Δ .+ (σΞ .^ 2) .+ (2 .* σΞ .* sqrt.(Δ))
        # p = 1
    elseif noise == :multiplicative
        D = Δ .* (1 .+ (Ξ .* sigma))
        # p = 1
    end
    return D
end

function stress(X, D)
    # Dx = pairwise(SqEuclidean(), X, dims=1)
    Dx = pairwise(Euclidean(), X, dims=1)
    return (Dx .- D) |> triu .|> (x -> x^2) |> mean
end