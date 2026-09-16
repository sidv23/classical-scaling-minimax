using Distances, LinearAlgebra, ProgressMeter, Random

# NOTE: `svd(A)` destructures via iteration as `(A.U, A.S, A.V)`, NOT
# `(A.U, A.S, A.Vt)` -- the third element is already A.V (so A ≈ U*Diagonal(S)*V'),

function slowprocrustes(X, X_ref)
    X_centered = X .- mean(X, dims=1)
    X_ref_centered = X_ref .- mean(X_ref, dims=1)

    U, _, V = svd(X_ref_centered' * X_centered)
    Q = V * U'
    translation = mean(X_ref, dims=1) - mean(X, dims=1) * Q
    return X * Q .+ translation
end

function procrustes(X, X_ref)
    X_centered = X .- mean(X, dims=1)
    X_ref_centered = X_ref .- mean(X_ref, dims=1)

    U, _, V = svd(X_ref_centered' * X_centered)
    Q = V * U'
    translation = mean(X_ref, dims=1) - mean(X, dims=1) * Q
    return X * Q .+ translation
end