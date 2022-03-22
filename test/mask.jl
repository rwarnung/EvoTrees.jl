using Statistics
using StatsBase: sample
using Revise
using EvoTrees
using BenchmarkTools

# prepare a dataset
features = rand(Int(1.25e6), 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1)) + 1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]


#############################
# CPU - linear
#############################
params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    mask = 1:90 => 1:3,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=64)

# asus laptopt: for 1.25e6 no eval: 9.650007 seconds (893.53 k allocations: 2.391 GiB, 5.52% gc time)
@time model = fit_evotree(params1, X_train, Y_train);
@btime model = fit_evotree($params1, $X_train, $Y_train);
@time pred_train = predict(model, X_train);
@btime pred_train = predict(model, X_train);
gain = importance(model, 1:100)

@time model, cache = EvoTrees.init_evotree(params1, X_train, Y_train);
@time EvoTrees.grow_evotree!(model, cache);
