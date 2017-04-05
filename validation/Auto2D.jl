module Auto2D

using AutomotiveDrivingModels
using AutoViz
using PDMats
using ForwardNets
using NGSIM
include("../pull_traces/multifeatureset.jl")

export SimParams, GaussianMLPDriver, LatentEncoder, tick, step

##################################
# SimParams
# include("../validation/load_policy.jl")

type SimState
    frame::Int
    start_frame::Int
    egoid::Int
    trajdata_index::Int

    scene::Scene
    rec::SceneRecord

    function SimState(context::IntegratedContinuous, rec_size::Int)
        retval = new()
        retval.scene = Scene()
        retval.rec = SceneRecord(rec_size, context.Δt)
        retval
    end
end

type SimParams
    context::IntegratedContinuous
    prime_history::Int
    ego_action_type::DataType

    trajdatas::Dict{Int, Trajdata}
    segments::Vector{TrajdataSegment}
    classes::Vector{Int}

    nsteps::Int
    step_counter::Int
    simstates::Vector{SimState}
    features::Vector{Float64}
    extractor::MultiFeatureExtractor
end

function SimParams(trajdatas::Dict{Int, Trajdata}, segments::Vector{TrajdataSegment},
    classes::Vector{Int},
    nsimstates::Int,
    prime_history::Int,
    nsteps::Int,
    ego_action_type::DataType,
    extractor::MultiFeatureExtractor,
    context = IntegratedContinuous(NGSIM_TIMESTEP,1),
    )

    simstates = Array(SimState, nsimstates)
    for i in 1 : length(simstates)
        simstates[i] = SimState(context, prime_history+1)
    end

    features = Array(Float64, length(extractor))

    SimParams(context, prime_history, ego_action_type, trajdatas, segments, classes, nsteps, 0, simstates, features, extractor)
end

##################################
# Gaussian MLP Driver

type GaussianMLPDriver{A<:DriveAction, F<:Real, G<:Real, H<:Real, E<:AbstractFeatureExtractor, M<:MvNormal} <: DriverModel{A, IntegratedContinuous}
    net::ForwardNet
    rec::SceneRecord
    pass::ForwardPass
    input::Vector{F}
    output::Vector{G}
    latent_state::Vector{H}
    driver_class::Vector{H}
    extractor::E
    mvnormal::M
    context::IntegratedContinuous
    use_latent::Bool
    oracle::Bool
end

_get_Σ_type{Σ,μ}(mvnormal::MvNormal{Σ,μ}) = Σ
function GaussianMLPDriver{A <: DriveAction}(::Type{A}, net::ForwardNet, extractor::AbstractFeatureExtractor, context::IntegratedContinuous;
    input::Symbol = :input,
    output::Symbol = :output,
    Σ::Union{Real, Vector{Float64}, Matrix{Float64},  Distributions.AbstractPDMat} = 0.1,
    rec::SceneRecord = SceneRecord(2, context.Δt),
    use_latent::Bool = false,
    oracle::Bool = false,
    )

    pass = calc_forwardpass(net, [input], [output])
    input_vec = net[input].tensor
    output = net[output].tensor
    latent_state = output  # Initialize to whatever
    driver_class = zeros(4)
    mvnormal = MvNormal(Array(Float64, 2), Σ)
    GaussianMLPDriver{A, eltype(input_vec), eltype(output), eltype(output), typeof(extractor), typeof(mvnormal)}(net, rec, pass, input_vec, output, latent_state, driver_class, extractor, mvnormal, context, use_latent, oracle)
end

AutomotiveDrivingModels.get_name(::GaussianMLPDriver) = "GaussianMLPDriver"
AutomotiveDrivingModels.action_context(model::GaussianMLPDriver) = model.context

# Set driver class to one-hot vector
function set_driver_class!{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P}, c::Int)
    model.driver_class = zeros(4)
    model.driver_class[c] = 1.0
end

# Set latent state at input to policy
function set_latent_state!{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P}, z::Vector{Float64})
    model.latent_state = z
end

# Empty record
function AutomotiveDrivingModels.reset_hidden_state!(model::GaussianMLPDriver)
    empty!(model.rec)
    model
end

# Fill in input with observations and latent samples
function AutomotiveDrivingModels.observe!{A,F,G,H,E,P}(
                                            model::GaussianMLPDriver{A,F,G,H,E,P}, 
                                            simparams::SimParams, 
                                            scene::Scene, 
                                            roadway::Roadway, 
                                            egoid::Int)

    update!(model.rec, scene)
    vehicle_index = get_index_of_first_vehicle_with_id(scene, egoid)
    o = pull_features!(simparams.extractor, simparams.features, model.rec, roadway, vehicle_index)
    o_norm = (o - model.extractor.feature_means)./model.extractor.feature_std
    
    if !(model.use_latent)
        if model.oracle && (:MLP_0 in keys(model.net.name_to_index))
            model.net[:MLP_0].input = cat(1, o_norm, model.driver_class)
        elseif model.oracle
            model.net[:LSTM_0].input = cat(1, o_norm, model.driver_class)
        elseif (:MLP_0 in keys(model.net.name_to_index))
            model.net[:MLP_0].input = o_norm
        elseif (:LSTM_0 in keys(model.net.name_to_index))
            model.net[:LSTM_0].input = o_norm
        else
            model.net[:GAIL_0].input = o_norm
        end
    else
        if (:MLP_0 in keys(model.net.name_to_index))
            model.net[:MLP_0].input = cat(1, o_norm, model.latent_state)
        else
            model.net[:LSTM_0].input = cat(1, o_norm, model.latent_state)
        end
    end
    forward!(model.pass)
    copy!(model.mvnormal.μ, model.output[1:2])

    model
end

function Base.rand{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P})
    a = rand(model.mvnormal)
    if model.extractor.norm_actions
        a = (a.*model.extractor.action_std + model.extractor.action_means)
    end
    convert(A, a)
end
Distributions.pdf{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P}, a::A) = pdf(model.mvnormal, convert(Vector{Float64}, a))
Distributions.logpdf{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P}, a::A) = logpdf(model.mvnormal, convert(Vector{Float64}, a))


##################################
# Latent Encoder
type LatentEncoder{F<:Real, G<:Real, E<:AbstractFeatureExtractor, M<:MvNormal}
    net::ForwardNet
    rec::SceneRecord
    pass::ForwardPass
    input::Vector{F}
    output::Vector{G}
    extractor::E
    mvnormal::M
    context::IntegratedContinuous
    z_dim::Int
end

_get_Σ_type{Σ,μ}(mvnormal::MvNormal{Σ,μ}) = Σ
function LatentEncoder(net::ForwardNet, extractor::AbstractFeatureExtractor, context::IntegratedContinuous;
    input::Symbol = :input,
    output::Symbol = :output,
    Σ::Union{Real, Vector{Float64}, Matrix{Float64},  Distributions.AbstractPDMat} = [0.1, 0.1],
    rec::SceneRecord = SceneRecord(2, context.Δt),
    z_dim::Int = 2,
    )

    pass = calc_forwardpass(net, [input], [output])
    input_vec = net[input].tensor
    output = net[output].tensor
    Σ = 0.1 * ones(z_dim)
    mvnormal = MvNormal(Array(Float64, z_dim), Σ)
    LatentEncoder{eltype(input_vec), eltype(output), typeof(extractor), typeof(mvnormal)}(net, rec, pass, input_vec, output, extractor, mvnormal, context, z_dim)
end

AutomotiveDrivingModels.get_name(::LatentEncoder) = "LatentEncoder"

function AutomotiveDrivingModels.reset_hidden_state!(model::LatentEncoder)
    empty!(model.rec)
    model
end

function AutomotiveDrivingModels.observe!{F,G,E,P}(
                                            model::LatentEncoder{F,G,E,P}, 
                                            simparams::SimParams, 
                                            scene::Scene, 
                                            roadway::Roadway, 
                                            egoid::Int)

    update!(model.rec, scene)
    vehicle_index = get_index_of_first_vehicle_with_id(scene, egoid)

    # Find and normalize features
    o = pull_features!(simparams.extractor, simparams.features, model.rec, roadway, vehicle_index)
    o_norm = (o - model.extractor.feature_means)./model.extractor.feature_std

    # Find and normalize vehicle actions
    a = zeros(2)
    a[1] = get(ACC, model.rec, roadway, vehicle_index, 0)
    a[2] = get(TURNRATEG, model.rec, roadway, vehicle_index, 0)
    a_norm = (a - model.extractor.action_means)./model.extractor.action_std

    model.net[:LSTM_0].input = cat(1, o_norm, a_norm)
    forward!(model.pass)
    copy!(model.mvnormal.μ, model.output[1:model.z_dim])
    copy!(model.mvnormal.Σ.diag, exp(model.output[(model.z_dim+1):2*model.z_dim]).^2)

    model
end
Base.rand{F,G,E,P}(model::LatentEncoder{F,G,E,P}) = rand(model.mvnormal)


########################################
# Propagate scenes

# Step scene forward
function step_forward!(simstate::SimState, simparams::SimParams, action_ego::Vector{Float64})

    trajdata = simparams.trajdatas[simstate.trajdata_index]
    veh_ego = get_vehicle(simstate.scene, simstate.egoid)
    action_ego = convert(simparams.ego_action_type, action_ego)

    # propagate the ego vehicle
    trajdata = simparams.trajdatas[simstate.trajdata_index]
    veh_ego = get_vehicle(simstate.scene, simstate.egoid)
    ego_state = propagate(veh_ego, action_ego, simparams.context, trajdata.roadway)

    simstate.frame += 1

    # pull new frame from trajdata
    get!(simstate.scene, trajdata, simstate.frame)

    # move in propagated ego vehicle
    veh_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)
    simstate.scene[veh_index].state = ego_state

    # update record
    update!(simstate.rec, simstate.scene)

    simstate
end


function tick(simparams::SimParams, u::Vector{Float64}, batch_index::Int=1)
    step_forward!(simparams.simstates[batch_index], simparams, u)
    simparams
end

function Base.step(simparams::SimParams, u::Vector{Float64}, batch_index::Int=1)
    tick(simparams, u, batch_index)
    simparams.step_counter += 1
end


end # module
