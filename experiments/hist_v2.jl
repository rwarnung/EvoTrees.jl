using Revise
using Statistics
using StatsBase: sample
using Base.Threads: @threads, @spawn
using BenchmarkTools
using StaticArrays

n_obs = Int(1e6)
n_vars = 100
n_bins = 64
K = 1
KK = 2 * K + 1
itot = collect(1:n_obs)
jtot = collect(1:n_vars)
x_bin = sample(UInt8.(1:n_bins), n_obs * n_vars);
x_bin = reshape(x_bin, n_obs, n_vars);

is = sample(itot, n_obs ÷ 2, replace=false, ordered=true)
js = sample(jtot, n_vars ÷ 2, replace=false, ordered=true)

"""
Current implementation (v0.13.0)
4.795 ms (73 allocations: 6.52 KiB)
"""
function hist_v0(hist, ∇, x_bin, is, js)
    @threads for j in js
        @inbounds @simd for i in is
            hid = 3 * x_bin[i, j] - 2
            hist[j][hid] += ∇[1, i]
            hist[j][hid+1] += ∇[2, i]
            hist[j][hid+2] += ∇[3, i]
        end
    end
end
hist = [zeros(KK * n_bins) for j in 1:n_vars];
∇ = rand(KK, n_obs)
@time hist_v0(hist, ∇, x_bin, is, js);
@btime hist_v0($hist, $∇, $x_bin, $is, $js);

"""
Current implementation (v0.13.0)
4.803 ms (73 allocations: 6.52 KiB)
"""
function hist_v0B(hist, ∇, x_bin, is, js)
    @threads for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[j][1, bin] += ∇[1, i]
            hist[j][2, bin] += ∇[2, i]
            hist[j][3, bin] += ∇[3, i]
        end
    end
end
hist = [zeros(KK, n_bins) for j in 1:n_vars];
∇ = rand(KK, n_obs);
@time hist_v0B(hist, ∇, x_bin, is, js);
@btime hist_v0B($hist, $∇, $x_bin, $is, $js);


"""
Build hist on Array instead of vec of vec
4.832 ms (73 allocations: 6.52 KiB)
"""
function hist_v1A(hist, ∇, x_bin, is, js)
    @threads for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[1, bin, j] += ∇[1, i]
            hist[2, bin, j] += ∇[2, i]
            hist[3, bin, j] += ∇[3, i]
        end
    end
end
hist = zeros(KK, n_bins, n_vars);
∇ = rand(KK, n_obs);
@time hist_v1A(hist, ∇, x_bin, is, js);
@btime hist_v1A($hist, $∇, $x_bin, $is, $js);

"""
Permute dims of ∇
4.639 ms (73 allocations: 6.52 KiB)
"""
function hist_v1B(hist, ∇, x_bin, is, js)
    @threads for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[1, bin, j] += ∇[i, 1]
            hist[2, bin, j] += ∇[i, 2]
            hist[3, bin, j] += ∇[i, 3]
        end
    end
end
hist = zeros(KK, n_bins, n_vars);
∇ = rand(n_obs, KK);
@time hist_v1B(hist, ∇, x_bin, is, js);
@btime hist_v1B($hist, $∇, $x_bin, $is, $js);


"""
Switch order of j and i loops
55.915 ms (73 allocations: 6.52 KiB)
"""
function hist_v1C(hist, ∇, x_bin, is, js)
    @threads for i in is
        @inbounds @simd for j in js
            bin = x_bin[i, j]
            hist[1, bin, j] += ∇[1, i]
            hist[2, bin, j] += ∇[2, i]
            hist[3, bin, j] += ∇[3, i]
        end
    end
end
hist = zeros(KK, n_bins, n_vars);
∇ = rand(KK, n_obs);
@time hist_v1C(hist, ∇, x_bin, is, js);
@btime hist_v1C($hist, $∇, $x_bin, $is, $js);


"""
@spawn
4.559 ms (314 allocations: 28.08 KiB)
"""
function hist_v1_spawn(hist, ∇, x_bin, is, js)
    @sync for j in js
        @spawn begin
            @inbounds @simd for i in is
                bin = x_bin[i, j]
                hist[1, bin, j] += ∇[1, i]
                hist[2, bin, j] += ∇[2, i]
                hist[3, bin, j] += ∇[3, i]
            end
        end
    end
end
hist = zeros(KK, n_bins, n_vars);
∇ = rand(KK, n_obs);
@time hist_v1_spawn(hist, ∇, x_bin, is, js);
@btime hist_v1_spawn($hist, $∇, $x_bin, $is, $js);


"""
Static Vector Matrix
4.297 ms (74 allocations: 6.55 KiB)
"""
function hist_v1_static(hist, ∇, x_bin, is, js)
    @threads for j in js
        @inbounds for i in is
            bin = x_bin[i, j]
            hist[bin, j] += ∇[i]
        end
    end
end
hist = zeros(SVector{KK,Float64}, n_bins, n_vars);
∇ = rand(SVector{KK,Float64}, n_obs);
@time hist_v1_static(hist, ∇, x_bin, is, js);
@btime hist_v1_static($hist, $∇, $x_bin, $is, $js);


"""
Static Vector Matrix - multi-thread js and is
56.924 ms (3813 allocations: 335.11 KiB)
"""
function hist_v1B_static(hist, ∇, x_bin, is, js)
    @threads for j in js
        @threads for i in is
            bin = x_bin[i, j]
            hist[bin, j] += ∇[i]
        end
    end
end
hist = zeros(SVector{KK,Float64}, n_bins, n_vars);
∇ = rand(SVector{KK,Float64}, n_obs);
@time hist_v1B_static(hist, ∇, x_bin, is, js);
@btime hist_v1B_static($hist, $∇, $x_bin, $is, $js);



"""
Build hist for all nodes of a given node

"""
function hist_v2_static(hist, ∇, x_bin, nidx, is, js)
    @threads for j in js
        @inbounds for i in is
            nid = nidx[i]
            bin = x_bin[i, j]
            hist[nid][bin, j] += ∇[i]
        end
    end
end
hist = [zeros(SVector{KK,Float64}, n_bins, n_vars) for nid in 1:31];
nidx = ones(UInt32, n_obs);
∇ = rand(SVector{KK,Float64}, n_obs);

@time hist_v2_static(hist, ∇, x_bin, nidx, is, js);
@btime hist_v2_static($hist, $∇, $x_bin, $nidx, $is, $js);


function iter_1(X_bin, hist, δ, 𝑖)
    hist .= 0.0
    @inbounds for i in 𝑖
        @inbounds for k in 1:3
            hist[k, X_bin[i, 1], 1] += δ[i, k]
        end
    end
end

𝑖_sample = sample(𝑖, Int(n_obs / 2), ordered=true)

@time iter_1(X_bin, hist, δ, 𝑖_sample)
@btime iter_1($X_bin, $hist, $δ, $𝑖_sample)



function iter_2(X_bin, hist, δ, 𝑖)
    hist .= 0.0
    @inbounds @simd for i in CartesianIndices(𝑖)
        @inbounds @simd for k in 1:3
            hist[k, X_bin[𝑖[i], 1], 1] += δ[𝑖[i], k]
        end
    end
end

𝑖_sample = sample(𝑖, Int(n_obs / 2), ordered=true)

@time iter_2(X_bin, hist, δ, 𝑖_sample)
@btime iter_2($X_bin, $hist, $δ, $𝑖_sample)



# slower
δ = rand(K, n_obs)
hist = zeros(K, n_bins, n_vars);
X_bin = sample(UInt8.(1:n_bins), n_obs * n_vars);
X_bin = reshape(X_bin, n_obs, n_vars);

function iter_1(X_bin, hist, δ, 𝑖)
    hist .= 0.0
    @inbounds for i in 𝑖
        @inbounds for k in 1:3
            hist[k, X_bin[i, 1], 1] += δ[k, i]
        end
    end
end

𝑖_sample = sample(𝑖, Int(n_obs / 2), ordered=true)

@time iter_1(X_bin, hist, δ, 𝑖_sample)
@btime iter_1($X_bin, $hist, $δ, $𝑖_sample)
