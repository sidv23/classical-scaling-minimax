using LinearAlgebra, Random, Pipe
using Distances, Distributions, Statistics

############################################################

function randBall(rng, n; d=2)
    X = randn(rng, n, d)
    X ./= sqrt.(sum(X .^ 2, dims=2))
    radii = rand(rng, n) .^ (1 / d)
    X .*= radii
    return X
end