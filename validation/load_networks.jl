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
                            input = :input, output = :output, Σ = Σ)
    else
        # extactor
        extractor = FeatureExtractor("./models/policy_vae_new.h5")
        Auto2D.LatentEncoder(net, extractor, IntegratedContinuous(0.1,1),
                            input = :input, output = :output)
    end
end

function _pull_W_b_h(filepath::AbstractString, path::AbstractString)
    W_xr = h5read(filepath, joinpath(path, "W_xr:0"))::Matrix{Float32}
    W_xu = h5read(filepath, joinpath(path, "W_xu:0"))::Matrix{Float32}
    W_xc = h5read(filepath, joinpath(path, "W_xc:0"))::Matrix{Float32}
    W_x = vcat(W_xr, W_xu, W_xc)
    
    W_hr = h5read(filepath, joinpath(path, "W_hr:0"))::Matrix{Float32}
    W_hu = h5read(filepath, joinpath(path, "W_hu:0"))::Matrix{Float32}
    W_hc = h5read(filepath, joinpath(path, "W_hc:0"))::Matrix{Float32}
    W_h = vcat(W_hr, W_hu, W_hc)
    
    b_r = h5read(filepath, joinpath(path, "b_r:0"))::Vector{Float32}
    b_u = h5read(filepath, joinpath(path, "b_u:0"))::Vector{Float32}
    b_c = h5read(filepath, joinpath(path, "b_c:0"))::Vector{Float32}
    b = vcat(b_r, b_u, b_c)
    
    (W_x, W_h, b)
end

function _pull_W_b_gail(filepath::AbstractString, path::AbstractString)
    W = h5read(filepath, joinpath(path, "W:0"))::Matrix{Float32}
    b = h5read(filepath, joinpath(path, "b:0"))::Vector{Float32}
    (W,b)
end

function load_gru_driver(
            filepath::AbstractString = POLICY_FILEPATH, 
            iteration::Int=-1; 
            gru_layer::Bool=true,
            bc_policy::Bool=false,
            action_type::DataType=AccelTurnrate)

    basepath = @sprintf("iter%05d/mlp_policy", iteration)
    layers = sort(collect(keys(h5read(filepath, basepath))))
    layers = layers[1:end-1]

    W, b = _pull_W_b_gail(filepath, joinpath(basepath, layers[1]))

    net = ForwardNet{Float32}()
    push!(net, Variable(:input, Array(Float32, size(W, 2))))

    # hidden layers
    i = 0
    for layer in layers
        layer_sym = Symbol("GAIL_" * string(i))
        W, b = _pull_W_b_gail(filepath, joinpath(basepath, layer))
        push!(net, Affine, layer_sym, lastindex(net), size(W, 1))
        copy!(net[layer_sym].W, W)
        copy!(net[layer_sym].b, b)
        push!(net, ELU, Symbol(layer*"ELU"), lastindex(net))
        i += 1
    end

    # Add GRU layer
    W, b = _pull_W_b_gail(filepath, joinpath(basepath, "output"))
    push!(net, Affine, :output_layer_mlp, lastindex(net), size(W, 1))
    copy!(net[:output_layer_mlp].W, W)
    copy!(net[:output_layer_mlp].b, b)
    push!(net, Variable(:output_mlp, output(net[:output_layer_mlp])), lastindex(net))
    push!(net, ELU, Symbol("output_ELU"), lastindex(net))

    basepath = @sprintf("iter%05d/gru_policy", iteration)
    W_x, W_h, b = _pull_W_b_h(filepath, joinpath(basepath, "mean_network", "gru"))
    push!(net, GRU, :gru, :output_mlp, size(W_h, 2))
    copy!(net[:gru].W_x, W_x)
    copy!(net[:gru].W_h, W_h)
    copy!(net[:gru].b, b)
    copy!(net[:gru].h_prev, zeros(size(W_h, 2)))
    push!(net, Variable(:output_gru, output(net[:gru])), lastindex(net))

    # Finally, output layer from GRU
    W, b = _pull_W_b_gail(filepath, joinpath(basepath, "mean_network", "output_flat"))
    push!(net, Affine, :output_layer, :gru, size(W, 1))
    copy!(net[:output_layer].W, W)
    copy!(net[:output_layer].b, b)
    push!(net, Variable(:output, output(net[:output_layer])), lastindex(net))

    logstdevs = vec(h5read(filepath, joinpath(basepath, "output_log_std/param:0")))::Vector{Float32}
    Σ = convert(Vector{Float64}, exp(logstdevs))
    Σ = Σ.^2

    # extactor
    extractor = FeatureExtractor(filepath, false)

    Auto2D.GaussianMLPDriver(action_type, net, extractor, IntegratedContinuous(0.1,1),
                        input = :input, output = :output, Σ = Σ)
end

