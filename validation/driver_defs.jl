using AutomotiveDrivingModels
include("Auto2D.jl")
include("load_networks.jl")
include("../pull_traces/multifeatureset.jl")

const TEST_ROADWAY = gen_stadium_roadway(3, length=250.0, radius=45.0)

const START_ROADINDS = [
    RoadIndex(proj(VecSE2(0.0,-1*DEFAULT_LANE_WIDTH,0.0), TEST_ROADWAY)),
    RoadIndex(proj(VecSE2(0.0,-0*DEFAULT_LANE_WIDTH,0.0), TEST_ROADWAY)),
    RoadIndex(proj(VecSE2(0.0,-2*DEFAULT_LANE_WIDTH,0.0), TEST_ROADWAY)),
    ]
# const START_DELTA_S = collect(10.0:8.0:18.0)
# const START_DELTA_S = collect(10.0:10.0:80.0)
const START_DELTA_S = [10.0]

function gen_standard(context, ncars)
    mlon = IntelligentDriverModel(
        σ=0.001,
        k_spd=1.5,
        T=0.25,
        s_min=1.0,
        a_max=2.5,
        d_cmf=2.5,
        )
    mlat = ProportionalLaneTracker(σ=0.001, kp=3.5, kd=2.5)
    mlane = MOBIL(context,
                  politeness=0.1,
                  advantage_threshold=0.01,
                 )

    model = Tim2DDriver(context, rec=SceneRecord(1, context.Δt, ncars), mlat=mlat, mlon=mlon, mlane=mlane)
    set_desired_speed!(model, 20.0)
    model
end
function gen_passive()
    filepath = "./models/policy_vae_new.h5"
    model = load_mlp_policy(filepath, encoder=true)
    Auto2D.set_latent_state!(model, [-1.0, 0.0])
    model
end
# More aggressive
function gen_aggressive()
    # println("aggressive")
    filepath = "./models/policy_vae_new.h5"
    model = load_mlp_policy(filepath, encoder=true)
    Auto2D.set_latent_state!(model, [2.0,0.0])
    model
end

function gen_medium1()
    filepath = "./models/policy_vae_new.h5"
    model = load_mlp_policy(filepath, encoder=true)
    Auto2D.set_latent_state!(model, [0.0,0.0])
    model
end

function gen_medium2()
    filepath = "./models/policy_vae_new.h5"
    model = load_mlp_policy(filepath, encoder=true)
    Auto2D.set_latent_state!(model, [1.5,0.0])
    model
end

function get_actions!{A<:DriveAction, D<:DriverModel}(
    actions::Vector{A},
    features::Vector{Float64},
    extractor::MultiFeatureExtractor,
    scene::Scene,
    roadway::Roadway,
    models::Dict{Int, D}, # id → model
    )


    for (i,veh) in enumerate(scene)
        model = models[veh.def.id]

        if :net in fieldnames(model)
            update!(model.rec, scene)
            vehicle_index = get_index_of_first_vehicle_with_id(scene, veh.def.id)
            o = pull_features!(extractor, features, model.rec, roadway, vehicle_index)
            o_norm = (o - model.extractor.feature_means)./model.extractor.feature_std
            model.net[:MLP_0].input = cat(1, o_norm, model.latent_state)
            forward!(model.pass)
            copy!(model.mvnormal.μ, model.output[1:2])
            
            # Sample action and clamp turnrate
            action = rand(model)
            actions[i] = convert(AccelTurnrate, [action.a, 0.0])

        else
            AutomotiveDrivingModels.observe!(model, scene, roadway, veh.def.id)
            actions[i] = rand(model)
        end
        
    end

    actions
end


function gen_start_state!(scene::Scene, context::IntegratedContinuous, models::Dict{Int, DriverModel};
    start_delta_s::Vector{Float64}=START_DELTA_S,
    start_roadinds::Vector{RoadIndex}=START_ROADINDS,
    base_speed::Float64 = 20.0,
    roadway::Roadway = TEST_ROADWAY,
    passive::Bool = true,
    aggressive::Bool = false,
    medium1::Bool = false,
    )

    x = 10
    j = 0
    id_count = 0

    empty!(scene)

    ncars = length(start_delta_s) * length(start_roadinds)

    for roadind in start_roadinds
        vehstate = VehicleState(Frenet(roadind, roadway), roadway, base_speed+0.5*randn())
        for delta_s in start_delta_s
            vehstate2 = move_along(vehstate, roadway, delta_s)
            vehdef = VehicleDef(id_count+=1, AgentClass.CAR, 4.826, 1.81)
            push!(scene, Vehicle(vehstate2, vehdef))

            # Initialize vehicle models
            if passive && id_count == 1
                models[id_count] = gen_passive()
            elseif aggressive && id_count == 1
                models[id_count] = gen_aggressive()
            elseif medium1 && id_count == 1
                models[id_count] = gen_medium1()
            elseif id_count == 1
                models[id_count] = gen_medium2()
            else
                models[id_count] = gen_standard(context, ncars)
            end
        end
    end
    scene
end