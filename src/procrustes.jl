using Distances, LinearAlgebra, ProgressMeter, Random


function slowprocrustes(X, X_ref)
    U, _, Vt = svd(X_ref' * X)
    Qt = Vt * U
    w = mean(X_ref, dims=1) - mean(X, dims=1) * Qt
    return w .+ (X * Qt)
end

function procrustes(X, X_ref)
    X_centered = X .- mean(X, dims=1)
    X_ref_centered = X_ref .- mean(X_ref, dims=1)

    U, _, Vt = svd(X_ref_centered' * X_centered)
    Q = Vt' * U'
    translation = mean(X_ref, dims=1) - mean(X, dims=1) * Q
    return X * Q .+ translation
end