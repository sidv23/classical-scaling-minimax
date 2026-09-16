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
    λ, V = eigs(E, nev=d, which=:LR)
    L = V .* .√λ'
    return L
end

function Dist(Δ, Ξ; sigma=1.0, noise=:additive)
    if noise == :additive
        D = Δ .+ (sigma .* Ξ)
    elseif noise == :additiveAbs
        D = Δ .+ sigma .* ((Ξ .^ 2) .+ (2 .* Ξ .* sqrt.(Δ)))
    elseif noise == :multiplicative
        D = Δ .* (1 .+ (sigma .* Ξ))
    end
    return D
end