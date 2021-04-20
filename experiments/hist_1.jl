using Statistics
using StatsBase:sample
using Base.Threads:@threads
using BenchmarkTools
using EvoTrees
using SIMD

n = Int(1e6)
nvars = 100
nbins = 64
𝑖 = collect(1:n)
𝑗 = collect(1:nvars)
δ = rand(n)
δ² = rand(n)
𝑤 = rand(n)

hist_δ = zeros(nbins, nvars)
hist_δ² = zeros(nbins, nvars)
hist_𝑤 = zeros(nbins, nvars)
X_bin = reshape(sample(UInt8.(1:nbins), n * nvars), n, nvars)

function iter_1(X_bin, hist_δ, δ, 𝑖, 𝑗)
    # hist_δ .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            hist_δ[X_bin[i,j], j] += δ[i]
        end
    end
end

@time iter_1(X_bin, hist_δ, δ, 𝑖, 𝑗)
@btime iter_1(X_bin, hist_δ, δ, 𝑖, 𝑗)

### 3 features in seperate hists
function iter_1B(X_bin, hist_δ, hist_δ², hist_𝑤, δ, δ², 𝑤, 𝑖, 𝑗)
    hist_δ .= 0.0
    hist_δ² .= 0.0
    hist_𝑤 .= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            hist_δ[X_bin[i,j], j] += δ[i]
            hist_δ²[X_bin[i,j], j] += δ²[i]
            hist_𝑤[X_bin[i,j], j] += 𝑤[i]
        end
    end
end

@time iter_1B(X_bin, hist_δ, hist_δ², hist_𝑤, δ, δ², 𝑤, 𝑖, 𝑗)
@btime iter_1B($X_bin, $hist_δ, $hist_δ², $hist_𝑤, $δ, $δ², $𝑤, $𝑖, $𝑗)


### 3 features in common hists
hist_δ𝑤 = zeros(3, nbins, nvars)
function iter_2(X_bin, hist_δ𝑤, δ, δ², 𝑤, 𝑖, 𝑗)
    hist_δ𝑤 .= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hist_δ𝑤[1, X_bin[i,j], j] += δ[i]
            hist_δ𝑤[2, X_bin[i,j], j] += δ²[i]
            hist_δ𝑤[3, X_bin[i,j], j] += 𝑤[i]
        end
    end
end

@time iter_2(X_bin, hist_δ𝑤, δ, δ², 𝑤, 𝑖, 𝑗)
@btime iter_2($X_bin, $hist_δ𝑤, $δ, $δ², $𝑤, $𝑖, $𝑗)

### 3 features in common hists - gradients/weight in single matrix
hist_δ𝑤 = zeros(3, nbins, nvars)
δ𝑤 = rand(3, n)

function iter_3(X_bin, hist_δ𝑤, δ𝑤, 𝑖, 𝑗)
    hist_δ𝑤 .= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hist_δ𝑤[1, X_bin[i,j], j] += δ𝑤[1,i]
            hist_δ𝑤[2, X_bin[i,j], j] += δ𝑤[2,i]
            hist_δ𝑤[3, X_bin[i,j], j] += δ𝑤[3,i]
        end
    end
end

@time iter_3(X_bin, hist_δ𝑤, δ𝑤, 𝑖, 𝑗)
@btime iter_3($X_bin, $hist_δ𝑤, $δ𝑤, $𝑖, $𝑗)


### 3 features in common hists - vector of matrix hists - gradients/weight in single matrix
hist_δ𝑤_vec = [zeros(3, nbins) for j in 1:nvars]
δ𝑤 = rand(3, n)

function iter_4(X_bin, hist_δ𝑤_vec, δ𝑤, 𝑖, 𝑗)    
    [hist_δ𝑤_vec[j] .= 0.0 for j in 𝑗]
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hist_δ𝑤_vec[j][1, X_bin[i,j]] += δ𝑤[1,i]
            hist_δ𝑤_vec[j][2, X_bin[i,j]] += δ𝑤[2,i]
            hist_δ𝑤_vec[j][3, X_bin[i,j]] += δ𝑤[3,i]
        end
    end
end

@time iter_4(X_bin, hist_δ𝑤_vec, δ𝑤, 𝑖, 𝑗)
@btime iter_4($X_bin, $hist_δ𝑤_vec, $δ𝑤, $𝑖, $𝑗)


### 3 features in common hists - vector of vec hists - gradients/weight in single vector
hist_δ𝑤_vec = [zeros(3 * nbins) for j in 1:nvars]
δ𝑤 = rand(3 * n)

function iter_5(X_bin, hist_δ𝑤_vec, δ𝑤, 𝑖, 𝑗)
    [hist_δ𝑤_vec[j] .= 0.0 for j in 𝑗]
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            id = 3 * i - 2
            hid = 3 * X_bin[i,j] - 2
            hist_δ𝑤_vec[j][hid] += δ𝑤[id]
            hist_δ𝑤_vec[j][hid + 1] += δ𝑤[id + 1]
            hist_δ𝑤_vec[j][hid + 2] += δ𝑤[id + 2]
        end
    end
end

@time iter_5(X_bin, hist_δ𝑤_vec, δ𝑤, 𝑖, 𝑗)
@btime iter_5($X_bin, $hist_δ𝑤_vec, $δ𝑤, $𝑖, $𝑗)


### 3 features in common hists - vector of vec hists - gradients/weight in single vector - explicit loop
hist_δ𝑤_vec = [zeros(3 * nbins) for j in 1:nvars]
δ𝑤 = rand(3 * n)

function iter_6(X_bin, hist_δ𝑤_vec, δ𝑤, 𝑖, 𝑗)
    [hist_δ𝑤_vec[j] .= 0.0 for j in 𝑗]
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            id = 3 * i - 2
            hid = 3 * X_bin[i,j] - 3
            for k in 1:3
                hist_δ𝑤_vec[j][hid + k] += δ𝑤[id + k]
            end
        end
    end
end

@time iter_6(X_bin, hist_δ𝑤_vec, δ𝑤, 𝑖, 𝑗)
@btime iter_6($X_bin, $hist_δ𝑤_vec, $δ𝑤, $𝑖, $𝑗)


### 3 features in common hists - vector of vec hists - gradients/weight in single vector - with node assignations
# hist_δ𝑤_vec = [[zeros(3 * nbins) for j in 1:nvars] for n in 1:16]
hist_δ𝑤_vec = [[zeros(3 * nbins) for n in 1:16] for j in 1:nvars]
δ𝑤 = rand(3 * n)
𝑛 = sample(1:16, n)

function iter_7(X_bin, hist_δ𝑤_vec::Vector{Vector{Vector{T}}}, δ𝑤::Vector{T}, 𝑖, 𝑗, 𝑛) where T
    # [hist_δ𝑤_vec[j][n] .= 0.0 for n in 𝑛]
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            id = 3 * i - 2
            hid = 3 * X_bin[i,j] - 2
            n = 𝑛[i]

            # hist_δ𝑤_vec[n][j][hid] += δ𝑤[id]
            # hist_δ𝑤_vec[n][j][hid + 1] += δ𝑤[id + 1]
            # hist_δ𝑤_vec[n][j][hid + 2] += δ𝑤[id + 2]

            hist_δ𝑤_vec[j][n][hid] += δ𝑤[id]
            hist_δ𝑤_vec[j][n][hid + 1] += δ𝑤[id + 1]
            hist_δ𝑤_vec[j][n][hid + 2] += δ𝑤[id + 2]
        end
    end
end

@time iter_7(X_bin, hist_δ𝑤_vec, δ𝑤, 𝑖, 𝑗, 𝑛)
@btime iter_7($X_bin, $hist_δ𝑤_vec, $δ𝑤, $𝑖, $𝑗, $𝑛)