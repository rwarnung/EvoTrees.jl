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
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

@btime edges = EvoTrees.get_edges(X_train, 64);
@btime x_bin = EvoTrees.binarize(X_train, edges);
@btime x_bin = apply_bins(X_train, edges);
@btime edges = prepare_bin_splits(X_train, 64);

function apply_bins(X, bin_splits)
  X_binned = zeros(UInt8, size(X))

  Threads.@threads for j in 1:size(X, 2)
    # for j in 1:feature_count(X)
    splits_for_feature = bin_splits[j]
    bin_count = length(splits_for_feature) + 1
    @inbounds for i in 1:size(X, 1)
      value = X[i, j]

      jump_step = div(bin_count - 1, 2)
      split_i = 1

      # Binary-ish jumping

      # invariant: split_i > 1 implies split_i split <= value
      while jump_step > 0
        while jump_step > 0 && splits_for_feature[split_i+jump_step] > value
          jump_step = div(jump_step, 2)
        end
        split_i += jump_step
        jump_step = div(jump_step, 2)
      end

      bin_i = bin_count
      for k in split_i:length(splits_for_feature)
        if splits_for_feature[k] > value
          bin_i = k
          break
        end
      end

      # split_i = findfirst(split_value -> split_value > value, @view splits_for_feature[split_i:length(splits_for_feature)])
      # bin_i   = split_i == nothing ? bin_count : split_i

      X_binned[i, j] = UInt8(bin_i) # Store as 1-255 to match Julia indexing. We leave 0 unused but saves us from having to remember to convert.
    end
  end

  X_binned
end


function prepare_bin_splits(X::Array{FeatureType,2}, bin_count=255) where {FeatureType<:AbstractFloat}
  if bin_count < 2 || bin_count > 255
    error("prepare_bin_splits: bin_count must be between 2 and 255")
  end
  ideal_sample_count = bin_count * 1_000
  is = sort(collect(Iterators.take(Random.shuffle(1:size(X, 1)), ideal_sample_count)))

  sample_count = length(is)
  split_count = bin_count - 1

  bin_splits = Vector{Vector{FeatureType}}(undef, size(X, 2))

  Threads.@threads for j in 1:size(X, 2)
    # for j in 1:feature_count(X)
    sorted_feature_values = sort(@view X[is, j])

    splits = zeros(eltype(sorted_feature_values), split_count)

    for split_i in 1:split_count
      split_sample_i = max(1, Int64(floor(sample_count / bin_count * split_i)))
      value_below_split = sorted_feature_values[split_sample_i]
      value_above_split = sorted_feature_values[min(split_sample_i + 1, sample_count)]
      splits[split_i] = (value_below_split + value_above_split) / 2.0f0 # Avoid coercing Float32 to Float64
    end

    bin_splits[j] = splits
  end

  bin_splits
end