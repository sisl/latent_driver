const PASSIVE_AGGRESSIVE_ROADWAY = gen_stadium_roadway(3, length=250.0, radius=45.0)

const START_ROADINDS = [
    RoadIndex(proj(VecSE2(0.0,-1*DEFAULT_LANE_WIDTH,0.0), PASSIVE_AGGRESSIVE_ROADWAY)),
    RoadIndex(proj(VecSE2(0.0,-0*DEFAULT_LANE_WIDTH,0.0), PASSIVE_AGGRESSIVE_ROADWAY)),
    RoadIndex(proj(VecSE2(0.0,-2*DEFAULT_LANE_WIDTH,0.0), PASSIVE_AGGRESSIVE_ROADWAY)),
    ]
const START_DELTA_S = collect(10.0:75.0:800.0)

function gen_aggressive(context, ncars)
    mlon = IntelligentDriverModel(
        σ=0.1,
        k_spd=1.5,
        T=0.25,
        s_min=1.0,
        a_max=5.0,
        d_cmf=5.0,
        )
    mlat = ProportionalLaneTracker(σ=0.1, kp=3.5, kd=2.5)
    mlane = MOBIL(context,
                  politeness=0.1,
                  advantage_threshold=0.01,
                 )

    model = Tim2DDriver(context, rec=SceneRecord(1, context.Δt, ncars), mlat=mlat, mlon=mlon, mlane=mlane)
    set_desired_speed!(model, 30.0 + randn())
    model
end
function gen_passive(context, ncars)
    mlon = IntelligentDriverModel(
        σ=0.1,
        k_spd=1.0,
        T=1.75,
        s_min=5.0,
        a_max=1.0,
        d_cmf=1.0,
        )
    mlat = ProportionalLaneTracker(σ=0.1, kp=3.0, kd=2.0)
    mlane = MOBIL(context,
                  politeness=0.5,
                  advantage_threshold=0.5,
                 )

    model = Tim2DDriver(context, rec=SceneRecord(1, context.Δt, ncars), mlat=mlat, mlon=mlon, mlane=mlane)
    set_desired_speed!(model, 10.0 + randn())
    model
end
# More aggressive
function gen_medium1(context, ncars)
    mlon = IntelligentDriverModel(
        σ=0.1,
        k_spd=1.0,
        T=0.25,
        s_min=1.0,
        a_max=1.0,
        d_cmf=1.0,
        )
    mlat = ProportionalLaneTracker(σ=0.1, kp=3.0, kd=2.0)
    mlane = MOBIL(context,
                  politeness=0.5,
                  advantage_threshold=0.5,
                 )

    model = Tim2DDriver(context, rec=SceneRecord(1, context.Δt, ncars), mlat=mlat, mlon=mlon, mlane=mlane)
    set_desired_speed!(model, 15.0 + randn())
    model
end

function gen_medium2(context, ncars)
    mlon = IntelligentDriverModel(
        σ=0.1,
        k_spd=1.5,
        T=1.75,
        s_min=5.0,
        a_max=5.0,
        d_cmf=5.0,
        )
    mlat = ProportionalLaneTracker(σ=0.1, kp=3.0, kd=2.0)
    mlane = MOBIL(context,
                  politeness=0.1,
                  advantage_threshold=0.01,
                 )

    model = Tim2DDriver(context, rec=SceneRecord(1, context.Δt, ncars), mlat=mlat, mlon=mlon, mlane=mlane)
    set_desired_speed!(model, 30.0 + randn())
    model
end

function gen_start_state!(scene::Scene, context::IntegratedContinuous, models::Dict{Int, DriverModel};
    start_delta_s::Vector{Float64}=START_DELTA_S,
    start_roadinds::Vector{RoadIndex}=START_ROADINDS,
    base_speed::Float64 = 20.0,
    roadway::Roadway = PASSIVE_AGGRESSIVE_ROADWAY,
    )

    x = 10
    j = 0
    id_count = 0
    carcolors = Dict{Int,Colorant}()
    car_class = Dict{Int,Int}()

    empty!(scene)

    ncars = length(start_delta_s) * length(start_roadinds)

    for roadind in start_roadinds
        vehstate = VehicleState(Frenet(roadind, roadway), roadway, base_speed+randn())
        for delta_s in start_delta_s
            vehstate2 = move_along(vehstate, roadway, delta_s)
            vehdef = VehicleDef(id_count+=1, AgentClass.CAR, 4.826, 1.81)
            push!(scene, Vehicle(vehstate2, vehdef))
            rand_num = rand()

            if rand_num <= 0.25
                models[id_count] = gen_passive(context, ncars)
                carcolors[id_count] = colorant"blue"
                car_class[id_count] = 1
            elseif (rand_num <= 0.5)
                models[id_count] = gen_medium2(context, ncars)
                carcolors[id_count] = colorant"cyan"
                car_class[id_count] = 2
            elseif (rand_num <= 0.75)
                models[id_count] = gen_medium1(context, ncars)
                carcolors[id_count] = colorant"green"
                car_class[id_count] = 3
            else
                models[id_count] = gen_aggressive(context, ncars)
                carcolors[id_count] = colorant"red"
                car_class[id_count] = 4
            end
            # models[id_count] = gen_medium1(context, ncars)
        end
    end

    scene, carcolors, car_class
end