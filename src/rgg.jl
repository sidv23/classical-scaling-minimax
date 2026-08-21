using LinearAlgebra, Random, Pipe
using Distances, Distributions, Statistics

hollow_transform(X, hollow) = @pipe map(
    x -> norm(x, Inf) >= hollow ?
         x :
         x .* (hollow + (norm(x, Inf) * (1 - hollow) / hollow)) ./ norm(x, Inf),
    eachrow(X)
) |> Matrix(hcat(_...)')

function randRect(n; scale=2, sigma=0.0, hollow=0.0, type=:additive, kwargs...)

    Unif = Uniform(-1, 1)

    if hollow > 1 - 1e-3 || hollow < 0.0
        @warn "hollow must be between 0 and 1. Setting it to zero."
        hollow = 0.0
    end
    # Distn = MixtureUniform(-1, )
    Xn = rand(Unif, n, 2)
    if hollow > 0.0
        Xn .= hollow_transform(Xn, hollow)
    end
    Xn .= Xn * [scale 0; 0 1/scale]
    # D = pairwise(SqEuclidean(), Xn, dims=1)
    D = pairwise(Euclidean(), Xn, dims=1)
    # A = [d ≤ R for d in D]
    if type == :additive
        ϵ = Symmetric(max.(rand(Unif, n, n) .* sigma, 0.0))
        ϵ[diagind(ϵ)] .= 0.0
        D = @. D + ϵ
    elseif type == :multiplicative
        ϵ = Symmetric(1 .+ max.(rand(Unif, n, n) .* sigma, -1.0))
        ϵ[diagind(ϵ)] .= 1.0
        D = @. D * sqrt(ϵ)
    end
    return Xn, D, ϵ
end

function randEllipse(n; scale=0.5, sigma=0.0, b=0.5, type=:additive, kwargs...)
    r = rand(n) .^ (2 / scale)
    t = rand(n) .* 2π
    Xn = [r[i] * sin(t[i] + π * j / 2) for i in eachindex(r), j in (1, 0)]
    Xn .= Xn * [1/b 0; 0 1]
    # D = pairwise(SqEuclidean(), Xn, dims=1)
    D = pairwise(Euclidean(), Xn, dims=1)
    # A = [d ≤ R for d in D]
    if type == :additive
        ϵ = Symmetric(max.(rand(Uniform(), n, n) .* sigma, 0.0))
        ϵ[diagind(ϵ)] .= 0.0
        D = @. D + ϵ
    elseif type == :multiplicative
        ϵ = Symmetric(1 .+ max.(rand(Uniform(), n, n) .* sigma, -1.0))
        ϵ[diagind(ϵ)] .= 1.0
        D = @. D * sqrt(ϵ)
    end
    return Xn, D, ϵ
end

############################################################
# Spheres
function randSphere(n::Int; d=2, sigma=0)
    signal = randn(n, d) |> (x -> x ./ norm.(eachrow(x)))
    noise = randn(n, d) .* sigma
    return signal .+ noise
end

function generate_centers(d)
    Q = eigen(((d + 1) / d) * I(d + 1) - (1 / d) * ones(d + 1, d + 1)).vectors
    return permutedims(Q'[2:end, :] .* sqrt((d + 1) / d))
end

function randBall(n; d=2)
    X = randn(n, d)
    X ./= sqrt.(sum(X .^ 2, dims=2))
    radii = rand(n) .^ (1 / d)
    X .*= radii
    return X
end
