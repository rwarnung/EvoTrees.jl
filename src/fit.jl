"""
    init_evotree(params::EvoTypes{T,U,S}, X::AbstractMatrix, Y::AbstractVector, W = nothing)
    
Initialise EvoTree
"""
function init_evotree(
    params::EvoTypes{L,T};
    x_train::AbstractMatrix,
    y_train::AbstractVector,
    w_train=nothing,
    offset_train=nothing,
    fnames=nothing
) where {L,T}

    levels = nothing
    x = convert(Matrix{T}, x_train)
    offset = !isnothing(offset_train) ? T.(offset_train) : nothing
    if L == Logistic
        K = 1
        y = T.(y_train)
        μ = [logit(mean(y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        K = 1
        y = T.(y_train)
        μ = fill(log(mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == Softmax
        if eltype(y_train) <: CategoricalValue
            levels = CategoricalArrays.levels(y_train)
            y = UInt32.(CategoricalArrays.levelcode.(y_train))
        else
            levels = sort(unique(y_train))
            yc = CategoricalVector(y_train, levels=levels)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
        K = length(levels)
        μ = zeros(T, K)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        K = 2
        y = T.(y_train)
        μ = [mean(y), log(std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == LogisticMLE
        K = 2
        y = T.(y_train)
        μ = [mean(y), log(std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        K = 1
        y = T.(y_train)
        μ = [mean(y)]
    end
    μ = T.(μ)

    # force a neutral bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)
    # initialize preds
    x_size = size(x)
    pred = zeros(T, K, x_size[1])
    @inbounds for i = 1:x_size[1]
        pred[:, i] .= μ
    end
    !isnothing(offset) && (pred .+= offset')

    # init EvoTree
    bias = [Tree{L,K,T}(μ)]
    fnames = isnothing(fnames) ? ["feat_$i" for i in axes(x, 2)] : string.(fnames)
    @assert length(fnames) == size(x, 2)
    info = Dict(:fnames => fnames, :levels => levels)
    m = EvoTree{L,K,T}(bias, info)

    # initialize gradients and weights
    δ𝑤 = zeros(T, 2 * K + 1, x_size[1])
    w = isnothing(w_train) ? ones(T, size(y)) : Vector{T}(w_train)
    @assert (length(y) == length(w) && minimum(w) > 0)
    δ𝑤[end, :] .= w

    # binarize data into quantiles
    edges = get_edges(x, params.nbins)
    x_bin = binarize(x, edges)

    𝑖_ = UInt32.(collect(1:x_size[1]))
    𝑗_ = UInt32.(collect(1:x_size[2]))
    𝑗 = zeros(eltype(𝑗_), ceil(Int, params.colsample * x_size[2]))
    nidx = ones(UInt32, x_size[1])

    # initialize histograms
    nodes = [TrainNode(x_size[2], params.nbins, K, T) for n = 1:2^params.max_depth-1]

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, x_size[2])
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    cache = (
        info=Dict(:nrounds => 0),
        x=x,
        y=y,
        K=K,
        nodes=nodes,
        pred=pred,
        𝑖_=𝑖_,
        𝑗_=𝑗_,
        𝑗=𝑗,
        nidx=nidx,
        δ𝑤=δ𝑤,
        edges=edges,
        x_bin=x_bin,
        monotone_constraints=monotone_constraints,
    )
    return m, cache
end


function grow_evotree!(evotree::EvoTree{L,K,T}, cache, params::EvoTypes{L,T}) where {L,K,T}

    # select random cols
    sample!(params.rng, cache.𝑗_, cache.𝑗, replace=false, ordered=true)

    # build a new tree
    update_grads!(cache.δ𝑤, cache.pred, cache.y, params)
    # assign a root and grow tree
    tree = Tree{L,K,T}(params.max_depth)
    grow_tree!(
        tree,
        cache.nodes,
        params,
        cache.δ𝑤,
        cache.edges,
        cache.𝑗,
        cache.out,
        cache.left,
        cache.right,
        cache.x_bin,
        cache.monotone_constraints,
        cache.nidx,
    )
    push!(evotree.trees, tree)
    predict!(cache.pred, tree, cache.x)
    cache[:info][:nrounds] += 1
    return nothing
end

# grow a single tree
function grow_tree!(
    tree::Tree{L,K,T},
    nodes::Vector{TrainNode{T}},
    params::EvoTypes{L,T},
    δ𝑤::Matrix{T},
    edges,
    𝑗,
    out,
    left,
    right,
    x_bin::AbstractMatrix,
    monotone_constraints,
    nidx
) where {L,K,T}

    # reset nodes
    @threads for n in nodes
        for j in 𝑗
            n.h[j] .= 0
        end
        n.∑ .= 0
        n.gain = 0
        n.gains .= 0
    end

    # reset
    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= @views vec(sum(δ𝑤[:, nodes[1].𝑖], dims=2))
    nodes[1].gain = get_gain(L, nodes[1].∑, params.lambda, K)
    # grow while there are remaining active nodes
    for depth = 1:params.max_depth
        update_hist!(L, nodes, nidx, δ𝑤, x_bin, 𝑗, K)
        for nid = 2^(depth-1):2^(depth)-1
            update_gains!(nodes[n], 𝑗, params, K, monotone_constraints)
            best = findmax(nodes[n].gains)
            if best[2][1] != params.nbins && best[1] > nodes[n].gain + params.gamma
                tree.gain[n] = best[1] - nodes[n].gain
                tree.cond_bin[n] = best[2][1]
                tree.feat[n] = best[2][2]
                tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
            end
            tree.split[n] = tree.cond_bin[n] != 0
            if !tree.split[n]
                pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
                # set to 0 the nidx of these obs
                nidx[nidx.==n] .= 0
            else
                nidx[nidx.==n] .= 0
                nodes[n<<1].𝑖, nodes[n<<1+1].𝑖 = _left, _right
                update_childs_∑!(L, nodes, n, best[2][1], best[2][2], K)
                nodes[n<<1].gain = get_gain(L, nodes[n<<1].∑, params.lambda, K)
                nodes[n<<1+1].gain = get_gain(L, nodes[n<<1+1].∑, params.lambda, K)

                if length(_right) >= length(_left)
                else
                end
            end
            # end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over active ids for a given depth
    return nothing
end


"""
    fit_evotree(params;
        x_train, y_train, w_train=nothing, offset_train=nothing;
        x_eval=nothing, y_eval=nothing, w_eval=nothing, offset_eval=nothing,
        early_stopping_rounds=9999,
        print_every_n=9999,
        verbosity=1)

Main training function. Performs model fitting given configuration `params`, `x_train`, `y_train` input data. 

# Arguments

- `params::EvoTypes`: configuration info providing hyper-paramters. `EvoTypes` comprises EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount and EvoTreeMLE.

# Keyword arguments

- `x_train::Matrix`: training data of size `[#observations, #features]`. 
- `y_train::Vector`: vector of train targets of length `#observations`.
- `w_train::Vector`: vector of train weights of length `#observations`. If `nothing`, a vector of ones is assumed.
- `offset_train::VecOrMat`: offset for the training data. Should match the size of the predictions.
- `x_eval::Matrix`: evaluation data of size `[#observations, #features]`. 
- `y_eval::Vector`: vector of evaluation targets of length `#observations`.
- `w_eval::Vector`: vector of evaluation weights of length `#observations`. Defaults to `nothing` (assumes a vector of 1s).
- `offset_eval::VecOrMat`: evaluation data offset. Should match the size of the predictions.
- `metric`: The evaluation metric that wil be tracked on `x_eval`, `y_eval` and optionally `w_eval` / `offset_eval` data. 
    Supported metrics are: 
    
        - `:mse`: mean-squared error. Adapted for general regression models.
        - `:rmse`: root-mean-squared error (CPU only). Adapted for general regression models.
        - `:mae`: mean absolute error. Adapted for general regression models.
        - `:logloss`: Adapted for `:logistic` regression models.
        - `:mlogloss`: Multi-class cross entropy. Adapted to `EvoTreeClassifier` classification models. 
        - `:poisson`: Poisson deviance. Adapted to `EvoTreeCount` count models.
        - `:gamma`: Gamma deviance. Adapted to regression problem on Gamma like, positively distributed targets.
        - `:tweedie`: Tweedie deviance. Adapted to regression problem on Tweedie like, positively distributed targets with probability mass at `y == 0`.
- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
- `fnames`: the names of the `x_train` features. If provided, should be a vector of string with `length(fnames) = size(x_train, 2)`.
"""
function fit_evotree(
    params::EvoTypes{L,T};
    x_train::AbstractMatrix,
    y_train::AbstractVector,
    w_train=nothing,
    offset_train=nothing,
    x_eval=nothing,
    y_eval=nothing,
    w_eval=nothing,
    offset_eval=nothing,
    metric=nothing,
    early_stopping_rounds=9999,
    print_every_n=9999,
    verbosity=1,
    fnames=nothing,
    return_logger=false
) where {L,T}

    # initialize model and cache
    if params.device == "gpu"
        m, cache = init_evotree_gpu(params; x_train, y_train, w_train, offset_train, fnames)
    else
        m, cache = init_evotree(params; x_train, y_train, w_train, offset_train, fnames)
    end

    # initialize callback and logger if tracking eval data
    logger = nothing
    if !isnothing(metric) && !isnothing(x_eval) && !isnothing(y_eval)
        cb = CallBack(params, m; x_eval, y_eval, w_eval, offset_eval, metric)
        logger = init_logger(;
            T,
            metric,
            maximise=is_maximise(cb.feval),
            early_stopping_rounds
        )
        cb(logger, 0, m.trees[end])
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    end

    # train loop
    for iter = 1:params.nrounds
        grow_evotree!(m, cache, params)
        if !isnothing(logger)
            cb(logger, iter, m.trees[end])
            if iter % print_every_n == 0 && verbosity > 0
                @info "iter $iter" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end # end of callback
    end
    if params.device == "gpu"
        GC.gc(true)
        CUDA.reclaim()
    end
    if return_logger
        return (m, logger)
    else
        return m
    end
end
