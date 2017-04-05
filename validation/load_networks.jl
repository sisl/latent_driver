using ForwardNets

# Create feature extractor
type FeatureExtractor <: AbstractFeatureExtractor
    feature_means::Vector{Float64} # for standardization
    feature_std::Vector{Float64} # for standardization
    action_means::Vector{Float64} # for standardization
    action_std::Vector{Float64} # for standardization
    norm_actions::Bool

    function FeatureExtractor(filepath::AbstractString, norm_actions::Bool=true)
        if norm_actions
            feature_means = vec(h5read(filepath, "policy/obs_mean"))
            feature_std = vec(h5read(filepath, "policy/obs_std"))
            action_means = vec(h5read(filepath, "policy/act_mean"))
            action_std = vec(h5read(filepath, "policy/act_std"))
        else
            feature_means = vec(h5read(filepath, "initial_obs_mean"))
            feature_std = vec(h5read(filepath, "initial_obs_std"))
            action_means = zeros(2)
            action_std = ones(2)
        end
        new(feature_means, feature_std, action_means, action_std, norm_actions)
    end
end
Base.length(::FeatureExtractor) = N_FEATURES
_standardize(v::Real, μ::Real, σ::Real) = (v - μ)/σ
function AutomotiveDrivingModels.pull_features!{F<:AbstractFloat}(ext::FeatureExtractor, features::Vector{F}, rec::SceneRecord, roadway::Roadway, vehicle_index::Int)
    pull_features!(features, rec, roadway, vehicle_index)

    # standardize
    for (i,v) in enumerate(features)
        features[i] = _standardize(v, ext.feature_means[i], ext.feature_std[i])
    end
    features
end

# Extract weights and biases
function _pull_W_b(filepath::AbstractString, path::AbstractString, layer::AbstractString)
    W = h5read(filepath, joinpath(path, "w_" * layer * ":0"))::Matrix{Float32}
    b = h5read(filepath, joinpath(path, "b_" * layer * ":0"))::Vector{Float32}
    (W,b)
end

# Load mlp policy into ForwardNets from h5 file
function load_mlp_policy(
            filepath::AbstractString = POLICY_FILEPATH, 
            iteration::Int=-1; 
            encoder::Bool=false,
            oracle::Bool=false,
            num_layers::Int=2,
            action_type::DataType=AccelTurnrate)

    # Extract first set of weights and biases, use to construct input
    basepath = "policy"
    W, b = _pull_W_b(filepath, basepath, string(0))
    net = ForwardNet{Float32}()
    push!(net, Variable(:input, Array(Float32, size(W, 2))))

    # hidden layers
    for i = 0:num_layers-1
        layer_sym = Symbol("MLP_" * string(i))
        W, b = _pull_W_b(filepath, basepath, string(i))
        push!(net, Affine, layer_sym, lastindex(net), size(W, 1))
        copy!(net[layer_sym].W, W)
        copy!(net[layer_sym].b, b)
        push!(net, ReLU, Symbol(string(i)*"ReLU"), lastindex(net))
    end

    # Output layer
    W, b = _pull_W_b(filepath, basepath, "end")
    push!(net, Affine, :output_layer, lastindex(net), size(W, 1))
    copy!(net[:output_layer].W, W)
    copy!(net[:output_layer].b, b)
    push!(net, Variable(:output, output(net[:output_layer])), lastindex(net))

    # Standard deviations for actions
    logstdevs = vec(h5read(filepath, joinpath(basepath, "a_logstd:0")))::Vector{Float32}
    Σ = convert(Vector{Float64}, exp(logstdevs))
    Σ = Σ.^2

    # extactor
    extractor = FeatureExtractor(filepath)

    # Construct model
    Auto2D.GaussianMLPDriver(action_type, net, extractor, IntegratedContinuous(0.1,1),
                    input = :input, output = :output, Σ = Σ, use_latent=encoder, oracle=oracle)

end

# Extract weights and biases for LSTM layer
function _pull_W_b_lstm(filepath::AbstractString, path::AbstractString, layer::AbstractString)
    W = h5read(filepath, joinpath(path, "Cell" * layer * "/LSTMCell/W_0:0"))::Matrix{Float32}
    b = h5read(filepath, joinpath(path, "Cell" * layer * "/LSTMCell/B:0"))::Vector{Float32}
    (W,b)
end

# Load lstm policy into ForwardNets from h5 file
function load_lstm(
            filepath::AbstractString = POLICY_FILEPATH, 
            iteration::Int=-1; 
            encoder::Bool=false,
            use_latent::Bool=false,
            oracle::Bool=false,
            num_layers::Int=2,
            action_type::DataType=AccelTurnrate)

    # Extract first set of weights and biases, use to construct input
    if encoder
        basepath = "encoder/rnn_decoder/MultiRNNCell"
    else
        basepath = "policy/rnn_decoder/MultiRNNCell"
    end
    W, b = _pull_W_b_lstm(filepath, basepath, string(0))

    # Find dimensions and add input
    H = convert(Int, size(W, 1)/4)
    D = size(W, 2) - H
    net = ForwardNet{Float32}()
    push!(net, Variable(:input, Array(Float32, D)))

    # hidden layers
    for i = 0:num_layers-1
        layer_sym = Symbol("LSTM_" * string(i))
        W, b = _pull_W_b_lstm(filepath, basepath, string(i))
        push!(net, LSTM, layer_sym, lastindex(net), H)
        copy!(net[layer_sym].W, W)
        copy!(net[layer_sym].b, b)
    end

    # Output layer
    if encoder
        W = h5read(filepath, "encoder/latent_w:0")::Matrix{Float32}
        b = h5read(filepath, "encoder/latent_b:0")::Vector{Float32}
    else
        W = h5read(filepath, "policy/lstm_w:0")::Matrix{Float32}
        b = h5read(filepath, "policy/lstm_b:0")::Vector{Float32}
    end
    push!(net, Affine, :output_layer, lastindex(net), size(W, 1))
    copy!(net[:output_layer].W, W)
    copy!(net[:output_layer].b, b)
    push!(net, Variable(:output, output(net[:output_layer])), lastindex(net))


    if !(encoder)
        # extactor
        extractor = FeatureExtractor(filepath)

        # Standard deviations for actions
        logstdevs = vec(h5read(filepath, "policy/a_logstd:0"))::Vector{Float32}
        Σ = convert(Vector{Float64}, exp(logstdevs))
        Σ = Σ.^2

        Auto2D.GaussianMLPDriver(action_type, net, extractor, IntegratedContinuous(0.1,1),
                            input = :input, output = :output, Σ = Σ, use_latent=use_latent, oracle=oracle)
    else
        # extactor
        z_dim = Int(size(W, 1)/2)
        extractor = FeatureExtractor("./models/policy_vae_new.h5")
        Auto2D.LatentEncoder(net, extractor, IntegratedContinuous(0.1,1),
                            input = :input, output = :output, z_dim = z_dim)
    end
end

