var documenterSearchIndex = {"docs":
[{"location":"api/#fit_evotree","page":"API","title":"fit_evotree","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"fit_evotree","category":"page"},{"location":"api/#EvoTrees.fit_evotree","page":"API","title":"EvoTrees.fit_evotree","text":"fit_evotree(params, X_train, Y_train, W_train=nothing;\n    X_eval=nothing, Y_eval=nothing, W_eval = nothing,\n    early_stopping_rounds=9999,\n    print_every_n=9999,\n    verbosity=1)\n\nMain training function. Performs model fitting given configuration params, X_train, Y_train input data. \n\nArguments\n\nparams::EvoTypes: configuration info providing hyper-paramters. EvoTypes comprises EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount or EvoTreeGaussian\nX_train::Matrix: training data of size [#observations, #features]. \nY_train::Vector: vector of train targets of length #observations.\nW_train::Vector: vector of train weights of length #observations. Defaults to nothing and a vector of ones is assumed.\n\nKeyword arguments\n\nX_eval::Matrix: evaluation data of size [#observations, #features]. \nY_eval::Vector: vector of evaluation targets of length #observations.\nW_eval::Vector: vector of evaluation weights of length #observations. Defaults to nothing (assumes a vector of 1s).\nearly_stopping_rounds::Integer: number of consecutive rounds without metric improvement after which fitting in stopped. \nprint_every_n: sets at which frequency logging info should be printed. \nverbosity: set to 1 to print logging info during training.\n\n\n\n\n\n","category":"function"},{"location":"api/#Predict","page":"API","title":"Predict","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"predict","category":"page"},{"location":"api/#MLJModelInterface.predict","page":"API","title":"MLJModelInterface.predict","text":"predict(loss::L, tree::Tree{T}, X::AbstractMatrix, K)\n\nPrediction from a single tree - assign each observation to its final leaf.\n\n\n\n\n\npredict(model::GBTree{T}, X::AbstractMatrix)\n\nPredictions from an EvoTrees model - sums the predictions from all trees composing the model.\n\n\n\n\n\n","category":"function"},{"location":"api/#Features-Importance","page":"API","title":"Features Importance","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"importance","category":"page"},{"location":"api/#EvoTrees.importance","page":"API","title":"EvoTrees.importance","text":"importance(model::GBTree, vars::AbstractVector)\n\nSorted normalized feature importance based on loss function gain.\n\n\n\n\n\n","category":"function"},{"location":"models/","page":"Models","title":"Models","text":"EvoTrees.jl supports four model families: ","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"- EvoTreeRegressor\n    - Linear (minimize mean-squared error)\n    - Logistic (minimize cross-entropy)\n    - L1: minimize mean-absolute error\n    - Quantile: minimize mean-absolute off the quantile\n- EvoTreeClassifier\n- EvoTreeCount\n- EvoTreeGaussian","category":"page"},{"location":"models/#EvoTreeRegressor","page":"Models","title":"EvoTreeRegressor","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"EvoTreeRegressor","category":"page"},{"location":"models/#EvoTrees.EvoTreeRegressor","page":"Models","title":"EvoTrees.EvoTreeRegressor","text":"EvoTreeRegressor(;kwargs...)\n\nA model type for constructing a EvoTreeRegressor, based on EvoTrees.jl, and implementing both an internal API the MLJ model interface. EvoTreeRegressor is used to perform the following regression types:\n\nlinear\nlogistic\nQuantile\nL1\n\nHyper-parameters\n\nloss=:linear:         One of :linear, :logistic, :quantile, :L1.\nnrounds=10:           Number of rounds. It corresponds to the number of trees that will be sequentially stacked.\nlambda::T=0.0:        L2 regularization term on weights. Must be >= 0. Higher lambda can result in a more robust model.\ngamma::T=0.0:         Minimum gain imprvement needed to perform a node split. Higher gamma can result in a more robust model.\nalpha::T=0.5:         Loss specific parameter in the [0-1] range:                            - :quantile: target quantile for the regression.                            - :L1: weighting parameters to positive vs negative residuals.                                   - Positive residual weights = alpha                                 - Negative residual weights = (1 - alpha)\nmax_depth=5:          Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.  A complete tree of depth N contains 2^(depth - 1) terminal leaves and 2^(depth - 1) - 1 split nodes. Compute cost is proportional to 2^max_depth. Typical optimal values are in the [3-9] range.\nmin_weight=0.0:       Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the weights vector.  \nrowsample=1.0:        Proportion of rows that are sampled at each iteration to build the tree. Should be ]0, 1].\ncolsample=1.0:        Proprtion of columns / features that are sampled at each iteration to build the tree. Should be ]0, 1].\nnbins=32:             Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins.\nrng=123:              Either an integer used as a seed to the random number generator or an actual random number generator (::Random.AbstractRNG). \nmetric::Symbol=:none: Metric that is to be tracked during the training process. One of: :none, :mse, :mae, :logloss.\ndevice=\"cpu\":         Hardware device to use for computations. Can be either \"cpu\" or \"gpu\". Only :linear and :logistic losses are supported on GPU.\n\nInternal API\n\nDo params = EvoTreeRegressor() to construct an instance with default hyper-parameters.  Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeRegressor(loss=...).\n\nTraining model\n\nA model is built using fit_evotree: \n\nmodel = fit_evotree(params, X_train, Y_train, W_train=nothing; kwargs...).\n\nInference\n\nPredictions are obtained using predict which returns a Matrix of size [nobs, 1]:\n\nEvoTrees.predict(model, X)\n\nMLJ Interface\n\nFrom MLJ, the type can be imported using:\n\nEvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees\n\nDo model = EvoTreeRegressor() to construct an instance with default hyper-parameters.  Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeRegressor(loss=...).\n\nTraining model\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y) where\n\nX: any table of input features (eg, a DataFrame) whose columns each have one of the following element scitypes: Continuous, Count, or <:OrderedFactor; check column scitypes with schema(X)\ny: is the target, which can be any AbstractVector whose element scitype is <:Continuous; check the scitype with scitype(y)\n\nTrain the machine using fit!(mach, rows=...).\n\nOperations\n\npredict(mach, Xnew): return predictions of the target given features Xnew having the same scitype as X above. Predictions are deterministic.\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\n:fitresult: The GBTree object returned by EvoTrees.jl fitting algorithm.\n\nReport\n\nThe fields of report(mach) are:\n\n:feature_importances: Feature importances based on the gain brought at each node split in the form of a Vector{Pair{String, Float64}}.  \n\nExamples\n\n# Internal API\nusing EvoTrees\nparams = EvoTreeRegressor(max_depth=5, nbins=32, nrounds=100)\nnobs, nfeats = 1_000, 5\nX, y = randn(nobs, nfeats), rand(nobs)\nmodel = fit_evotree(params, X, y)\npreds = EvoTrees.predict(model, X)\n\n# MLJ Interface\nusing MLJ\nEvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees\nmodel = EvoTreeRegressor(max_depth=5, nbins=32, nrounds=100)\nX, y = @load_boston\nmach = machine(model, X, y) |> fit!\npreds = predict(mach, X)\n\n\n\n\n\n","category":"type"},{"location":"models/#EvoTreeClassifier","page":"Models","title":"EvoTreeClassifier","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"EvoTreeClassifier","category":"page"},{"location":"models/#EvoTrees.EvoTreeClassifier","page":"Models","title":"EvoTrees.EvoTreeClassifier","text":"EvoTreeClassifier(;kwargs...)\n\nA model type for constructing a EvoTreeClassifier, based on EvoTrees.jl, and implementing both an internal API the MLJ model interface. EvoTreeClassifier is used to perform multi-class classification, using cross-entropy loss.\n\nHyper-parameters\n\nloss::Symbol=:softmax:      Fixed to softmax by default.\nnrounds=10:                 Number of rounds. It corresponds to the number of trees that will be sequentially stacked.\nlambda::T=0.0:              L2 regularization term on weights. Must be >= 0. Higher lambda can result in a more robust model.\ngamma::T=0.0:               Minimum gain imprvement needed to perform a node split. Higher gamma can result in a more robust model.\nmax_depth=5:                Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.  A complete tree of depth N contains 2^(depth - 1) terminal leaves and 2^(depth - 1) - 1 split nodes. Compute cost is proportional to 2^max_depth. Typical optimal values are in the [3-9] range.\nmin_weight=0.0:             Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the weights vector.  \nrowsample=1.0:              Proportion of rows that are sampled at each iteration to build the tree. Should be ]0, 1].\ncolsample=1.0:              Proprtion of columns / features that are sampled at each iteration to build the tree. Should be ]0, 1].\nnbins=32:                   Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins.\nrng=123:                    Either an integer used as a seed to the random number generator or an actual random number generator (::Random.AbstractRNG). \nmetric::Symbol=:none:       Metric that is to be tracked during the training process. One of: :none, :mlogloss.\ndevice=\"cpu\":               Hardware device to use for computations. Only CPU is supported at the moment.\n\nInternal API\n\nDo params = EvoTreeClassifier() to construct an instance with default hyper-parameters.  Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeClassifier(max_depth=...).\n\nTraining model\n\nA model is built using fit_evotree: \n\nmodel = fit_evotree(params, X_train, Y_train, W_train=nothing; kwargs...).\n\nInference\n\nPredictions are obtained using predict which returns a Matrix of size [nobs, K] where K is the number of classes:\n\nEvoTrees.predict(model, X)\n\nMLJ\n\nFrom MLJ, the type can be imported using:\n\nEvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees\n\nDo model = EvoTreeClassifier() to construct an instance with default hyper-parameters.  Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeClassifier(loss=...).\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y) where\n\nX: any table of input features (eg, a DataFrame) whose columns each have one of the following element scitypes: Continuous, Count, or <:OrderedFactor; check column scitypes with schema(X)\ny: is the target, which can be any AbstractVector whose element scitype is <:Finite; check the scitype with scitype(y)\n\nTrain the machine using fit!(mach, rows=...).\n\nOperations\n\npredict(mach, Xnew): return predictions of the target given features Xnew having the same scitype as X above.  Predictions are probabilistic.\npredict_mode(mach, Xnew): returns the mode of each of the prediction above.\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\n:fitresult: The GBTree object returned by EvoTrees.jl fitting algorithm.\n\nReport\n\nThe fields of report(mach) are:\n\n:feature_importances: Feature importances based on the gain brought at each node split in the form of a Vector{Pair{String, Float64}}.  \n\nExamples\n\n# Internal API\nusing EvoTrees\nparams = EvoTreeClassifier(max_depth=5, nbins=32, nrounds=100)\nnobs, nfeats = 1_000, 5\nX, y = randn(nobs, nfeats), rand(1:3, nobs)\nmodel = fit_evotree(params, X, y)\npreds = EvoTrees.predict(model, X)\n\n# MLJ Interface\nusing MLJ\nEvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees\nmodel = EvoTreeClassifier(max_depth=5, nbins=32, nrounds=100)\nX, y = @load_iris\nmach = machine(model, X, y) |> fit!\npreds = predict(mach, X)\npreds = predict_mode(mach, X)\n\nSee also EvoTrees.jl.\n\n\n\n\n\n","category":"type"},{"location":"models/#EvoTreeCount","page":"Models","title":"EvoTreeCount","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"EvoTreeCount","category":"page"},{"location":"models/#EvoTrees.EvoTreeCount","page":"Models","title":"EvoTrees.EvoTreeCount","text":"EvoTreeCount(;kwargs...)\n\nA model type for constructing a EvoTreeCount, based on EvoTrees.jl, and implementing both an internal API the MLJ model interface. EvoTreeCount is used to perform Poisson probabilistic regression on count target.\n\nHyper-parameters\n\nloss::Symbol=:poisson:      Fixed to poisson by default.\nnrounds=10:                 Number of rounds. It corresponds to the number of trees that will be sequentially stacked.\nlambda::T=0.0:              L2 regularization term on weights. Must be >= 0. Higher lambda can result in a more robust model.\ngamma::T=0.0:               Minimum gain imprvement needed to perform a node split. Higher gamma can result in a more robust model.\nmax_depth=5:                Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.  A complete tree of depth N contains 2^(depth - 1) terminal leaves and 2^(depth - 1) - 1 split nodes. Compute cost is proportional to 2^max_depth. Typical optimal values are in the [3-9] range.\nmin_weight=0.0:             Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the weights vector.  \nrowsample=1.0:              Proportion of rows that are sampled at each iteration to build the tree. Should be ]0, 1].\ncolsample=1.0:              Proprtion of columns / features that are sampled at each iteration to build the tree. Should be ]0, 1].\nnbins=32:                   Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins.\nrng=123:                    Either an integer used as a seed to the random number generator or an actual random number generator (::Random.AbstractRNG). \nmetric::Symbol=:none:       Metric that is to be tracked during the training process. One of: :none, :poisson, :mae, :mse.\ndevice=\"cpu\":               Hardware device to use for computations. Only CPU is supported at the moment.\n\nInternal API\n\nDo params = EvoTreeCount() to construct an instance with default hyper-parameters.  Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeCount(max_depth=...).\n\nTraining model\n\nA model is built using fit_evotree: \n\nmodel = fit_evotree(params, X_train, Y_train, W_train=nothing; kwargs...).\n\nInference\n\nPredictions are obtained using predict which returns a Matrix of size [nobs, 1]:\n\nEvoTrees.predict(model, X)\n\nMLJ\n\nFrom MLJ, the type can be imported using:\n\nEvoTreeCount = @load EvoTreeCount pkg=EvoTrees\n\nDo model = EvoTreeCount() to construct an instance with default hyper-parameters.  Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeCount(loss=...).\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y) where\n\nX: any table of input features (eg, a DataFrame) whose columns each have one of the following element scitypes: Continuous, Count, or <:OrderedFactor; check column scitypes with schema(X)\ny: is the target, which can be any AbstractVector whose element scitype is <:Count; check the scitype with scitype(y)\n\nTrain the machine using fit!(mach, rows=...).\n\nOperations\n\npredict(mach, Xnew): returns the Poisson distribution given features Xnew having the same scitype as X above. \n\nPredictions are probabilistic.\n\nSpecific metrics can also be predicted using:\n\npredict_mean(mach, Xnew)\npredict_mode(mach, Xnew)\npredict_median(mach, Xnew)\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\n:fitresult: The GBTree object returned by EvoTrees.jl fitting algorithm.\n\nReport\n\nThe fields of report(mach) are:\n\n:feature_importances: Feature importances based on the gain brought at each node split in the form of a Vector{Pair{String, Float64}}.  \n\nExamples\n\n# Internal API\nusing EvoTrees\nparams = EvoTreeCount(max_depth=5, nbins=32, nrounds=100)\nnobs, nfeats = 1_000, 5\nX, y = randn(nobs, nfeats), rand(0:2, nobs)\nmodel = fit_evotree(params, X, y)\npreds = EvoTrees.predict(model, X)\n\nusing MLJ\nEvoTreeCount = @load EvoTreeCount pkg=EvoTrees\nmodel = EvoTreeCount(max_depth=5, nbins=32, nrounds=100)\nnobs, nfeats = 1_000, 5\nX, y = randn(nobs, nfeats), rand(0:2, nobs)\nmach = machine(model, X, y) |> fit!\npreds = predict(mach, X)\npreds = predict_mean(mach, X)\npreds = predict_mode(mach, X)\npreds = predict_median(mach, X)\n\n\nSee also EvoTrees.jl.\n\n\n\n\n\n","category":"type"},{"location":"models/#EvoTreeGaussian","page":"Models","title":"EvoTreeGaussian","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"EvoTreeGaussian","category":"page"},{"location":"models/#EvoTrees.EvoTreeGaussian","page":"Models","title":"EvoTrees.EvoTreeGaussian","text":"EvoTreeGaussian(;kwargs...)\n\nA model type for constructing a EvoTreeGaussian, based on EvoTrees.jl, and implementing both an internal API the MLJ model interface. EvoTreeGaussian is used to perform Gaussain probabilistic regression, fitting μ and σ parameters to maximize likelihood.\n\nHyper-parameters\n\nloss::Symbol=:gaussian:     Fixed to gaussian by default.\nnrounds=10:                 Number of rounds. It corresponds to the number of trees that will be sequentially stacked.\nlambda::T=0.0:              L2 regularization term on weights. Must be >= 0. Higher lambda can result in a more robust model.\ngamma::T=0.0:               Minimum gain imprvement needed to perform a node split. Higher gamma can result in a more robust model.\nmax_depth=5:                Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.  A complete tree of depth N contains 2^(depth - 1) terminal leaves and 2^(depth - 1) - 1 split nodes. Compute cost is proportional to 2^max_depth. Typical optimal values are in the [3-9] range.\nmin_weight=0.0:             Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the weights vector.  \nrowsample=1.0:              Proportion of rows that are sampled at each iteration to build the tree. Should be ]0, 1].\ncolsample=1.0:              Proprtion of columns / features that are sampled at each iteration to build the tree. Should be ]0, 1].\nnbins=32:                   Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins.\nrng=123:                    Either an integer used as a seed to the random number generator or an actual random number generator (::Random.AbstractRNG). \nmetric::Symbol=:none:       Metric that is to be tracked during the training process. One of: :none, :gaussian.\ndevice=\"cpu\":               Hardware device to use for computations. Only CPU is supported at the moment.\n\nInternal API\n\nDo params = EvoTreeGaussian() to construct an instance with default hyper-parameters.  Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeGaussian(max_depth=...).\n\nTraining model\n\nA model is built using fit_evotree: \n\nfit_evotree(params, X_train, Y_train, W_train=nothing; kwargs...).\n\nInference\n\nPredictions are obtained using predict which returns a Matrix of size [nobs, 2] where the second dimensions refer to μ and σ respectively:\n\nEvoTrees.predict(model, X)\n\nMLJ\n\nFrom MLJ, the type can be imported using:\n\nEvoTreeGaussian = @load EvoTreeGaussian pkg=EvoTrees\n\nDo model = EvoTreeGaussian() to construct an instance with default hyper-parameters.  Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeGaussian(loss=...).\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y) where\n\nX: any table of input features (eg, a DataFrame) whose columns each have one of the following element scitypes: Continuous, Count, or <:OrderedFactor; check column scitypes with schema(X)\ny: is the target, which can be any AbstractVector whose element scitype is <:Continuous; check the scitype with scitype(y)\n\nTrain the machine using fit!(mach, rows=...).\n\nOperations\n\npredict(mach, Xnew): returns the Gaussian distribution given features Xnew having the same scitype as X above. \n\nPredictions are probabilistic.\n\nSpecific metrics can also be predicted using:\n\npredict_mean(mach, Xnew)\npredict_mode(mach, Xnew)\npredict_median(mach, Xnew)\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\n:fitresult: The GBTree object returned by EvoTrees.jl fitting algorithm.\n\nReport\n\nThe fields of report(mach) are:\n\n:feature_importances: Feature importances based on the gain brought at each node split in the form of a Vector{Pair{String, Float64}}.  \n\nExamples\n\n# Internal API\nusing EvoTrees\nparams = EvoTreeGaussian(max_depth=5, nbins=32, nrounds=100)\nnobs, nfeats = 1_000, 5\nX, y = randn(nobs, nfeats), rand(nobs)\nmodel = fit_evotree(params, X, y)\npreds = EvoTrees.predict(model, X)\n\n# MLJ Interface\nusing MLJ\nEvoTreeGaussian = @load EvoTreeGaussian pkg=EvoTrees\nmodel = EvoTreeGaussian(max_depth=5, nbins=32, nrounds=100)\nX, y = @load_boston\nmach = machine(model, X, y) |> fit!\npreds = predict(mach, X)\npreds = predict_mean(mach, X)\npreds = predict_mode(mach, X)\npreds = predict_median(mach, X)\n\n\n\n\n\n","category":"type"},{"location":"examples-MLJ/#MLJ-Integration","page":"Examples - MLJ","title":"MLJ Integration","text":"","category":"section"},{"location":"examples-MLJ/","page":"Examples - MLJ","title":"Examples - MLJ","text":"EvoTrees.jl provides a first-class integration with the MLJ ecosystem. ","category":"page"},{"location":"examples-MLJ/","page":"Examples - MLJ","title":"Examples - MLJ","text":"See official project page for more info.","category":"page"},{"location":"examples-MLJ/","page":"Examples - MLJ","title":"Examples - MLJ","text":"To use with MLJ, an EvoTrees model must first be initialized using either EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount or EvoTreeGaussian. The model is then passed to MLJ's machine, opening access to the rest of the MLJ modeling ecosystem. ","category":"page"},{"location":"examples-MLJ/","page":"Examples - MLJ","title":"Examples - MLJ","text":"using StatsBase: sample\nusing EvoTrees\nusing EvoTrees: sigmoid, logit # only needed to create the synthetic data below\nusing MLJBase\n\nfeatures = rand(10_000) .* 5 .- 2\nX = reshape(features, (size(features)[1], 1))\nY = sin.(features) .* 0.5 .+ 0.5\nY = logit(Y) + randn(size(Y))\nY = sigmoid(Y)\ny = Y\nX = MLJBase.table(X)\n\n# linear regression\ntree_model = EvoTreeRegressor(loss=:linear, max_depth=5, eta=0.05, nrounds=10)\n\n# set machine\nmach = machine(tree_model, X, y)\n\n# partition data\ntrain, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split\n\n# fit data\nfit!(mach, rows=train, verbosity=1)\n\n# continue training\nmach.model.nrounds += 10\nfit!(mach, rows=train, verbosity=1)\n\n# predict on train data\npred_train = predict(mach, selectrows(X, train))\nmean(abs.(pred_train - selectrows(Y, train)))\n\n# predict on test data\npred_test = predict(mach, selectrows(X, test))\nmean(abs.(pred_test - selectrows(Y, test)))","category":"page"},{"location":"examples-API/#Regression","page":"Examples - API","title":"Regression","text":"","category":"section"},{"location":"examples-API/","page":"Examples - API","title":"Examples - API","text":"Minimal example to fit a noisy sinus wave.","category":"page"},{"location":"examples-API/","page":"Examples - API","title":"Examples - API","text":"(Image: )","category":"page"},{"location":"examples-API/","page":"Examples - API","title":"Examples - API","text":"using EvoTrees\nusing EvoTrees: sigmoid, logit\n\n# prepare a dataset\nfeatures = rand(10000) .* 20 .- 10\nX = reshape(features, (size(features)[1], 1))\nY = sin.(features) .* 0.5 .+ 0.5\nY = logit(Y) + randn(size(Y))\nY = sigmoid(Y)\n𝑖 = collect(1:size(X, 1))\n\n# train-eval split\n𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)\ntrain_size = 0.8\n𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]\n𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]\n\nX_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]\nY_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]\n\nparams1 = EvoTreeRegressor(\n    loss=:linear, metric=:mse,\n    nrounds=100, nbins = 100,\n    lambda = 0.5, gamma=0.1, eta=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\n\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_eval_linear = predict(model, X_eval)\n\n# logistic / cross-entropy\nparams1 = EvoTreeRegressor(\n    loss=:logistic, metric = :logloss,\n    nrounds=100, nbins = 100,\n    lambda = 0.5, gamma=0.1, eta=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\n\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_eval_logistic = predict(model, X_eval)\n\n# L1\nparams1 = EvoTreeRegressor(\n    loss=:L1, alpha=0.5, metric = :mae,\n    nrounds=100, nbins=100,\n    lambda = 0.5, gamma=0.0, eta=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\n\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_eval_L1 = predict(model, X_eval)","category":"page"},{"location":"examples-API/#Poisson-Count","page":"Examples - API","title":"Poisson Count","text":"","category":"section"},{"location":"examples-API/","page":"Examples - API","title":"Examples - API","text":"# Poisson\nparams1 = EvoTreeCount(\n    loss=:poisson, metric = :poisson,\n    nrounds=100, nbins = 100,\n    lambda = 0.5, gamma=0.1, eta=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\n\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_eval_poisson = predict(model, X_eval)","category":"page"},{"location":"examples-API/#Quantile-Regression","page":"Examples - API","title":"Quantile Regression","text":"","category":"section"},{"location":"examples-API/","page":"Examples - API","title":"Examples - API","text":"(Image: )","category":"page"},{"location":"examples-API/","page":"Examples - API","title":"Examples - API","text":"# q50\nparams1 = EvoTreeRegressor(\n    loss=:quantile, alpha=0.5, metric = :quantile,\n    nrounds=200, nbins = 100,\n    lambda = 0.1, gamma=0.0, eta=0.05,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\n\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_train_q50 = predict(model, X_train)\n\n# q20\nparams1 = EvoTreeRegressor(\n    loss=:quantile, alpha=0.2, metric = :quantile,\n    nrounds=200, nbins = 100,\n    lambda = 0.1, gamma=0.0, eta=0.05,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\n\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_train_q20 = predict(model, X_train)\n\n# q80\nparams1 = EvoTreeRegressor(\n    loss=:quantile, alpha=0.8,\n    nrounds=200, nbins = 100,\n    lambda = 0.1, gamma=0.0, eta=0.05,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\n\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_train_q80 = predict(model, X_train)","category":"page"},{"location":"examples-API/#Gaussian-Max-Likelihood","page":"Examples - API","title":"Gaussian Max Likelihood","text":"","category":"section"},{"location":"examples-API/","page":"Examples - API","title":"Examples - API","text":"(Image: )","category":"page"},{"location":"examples-API/","page":"Examples - API","title":"Examples - API","text":"params1 = EvoTreeGaussian(\n    loss=:gaussian, metric=:gaussian,\n    nrounds=100, nbins=100,\n    lambda = 0.0, gamma=0.0, eta=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0, seed=123)","category":"page"},{"location":"#[EvoTress.jl](https://github.com/Evovest/EvoTrees.jl)","page":"Introduction","title":"EvoTress.jl","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"A Julia implementation of boosted trees with CPU and GPU support. Efficient histogram based algorithms with support for multiple loss functions, including various regressions, multi-classification and Gaussian max likelihood. ","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"See the examples-API section to get started using the internal API, or examples-MLJ to use within the MLJ framework.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Complete details about hyper-parameters are found in the Models section.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"R binding available.","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Latest:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"julia> Pkg.add(\"https://github.com/Evovest/EvoTrees.jl\")","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"From General Registry:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"julia> Pkg.add(\"EvoTrees\")","category":"page"}]
}
