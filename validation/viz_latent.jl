using AutomotiveDrivingModels
using NGSIM
using HDF5, JLD
using AutoViz
using Reel
using PGFPlots

include("driver_defs.jl")

srand(0)

# generate roadway
roadway = TEST_ROADWAY

# starting locations
start_roadinds = START_ROADINDS
start_delta_s = START_DELTA_S
ncars = length(start_roadinds) * length(start_delta_s)
nframes_per_run = ceil(Int, 5.0 / NGSIM_TIMESTEP) # 30 seconds of driving
nruns = 400
nstates_total = ncars*nruns*nframes_per_run
ntrain_frames = ncars*nruns*(nframes_per_run-1)

println("ncars: ", ncars)
println("nframes_per_run: ", nframes_per_run)
println("nruns: ", nruns)
println("nstates_total: ", nstates_total)
println("ntrain_frames: ", ntrain_frames)

const EXTRACT_CORE = true
const EXTRACT_TEMPORAL = false
const EXTRACT_WELL_BEHAVED = true
const EXTRACT_NEIGHBOR_FEATURES = false
const EXTRACT_CARLIDAR_RANGERATE = true
const CARLIDAR_NBEAMS = 20
const ROADLIDAR_NBEAMS = 0
const ROADLIDAR_NLANES = 2
const N_FEATURES = 51
extractor = MultiFeatureExtractor(EXTRACT_CORE, EXTRACT_TEMPORAL, 
                                    EXTRACT_WELL_BEHAVED, EXTRACT_NEIGHBOR_FEATURES, 
                                    EXTRACT_CARLIDAR_RANGERATE, CARLIDAR_NBEAMS,
                                    ROADLIDAR_NBEAMS, ROADLIDAR_NLANES)
features = Array(Float64, N_FEATURES)
# featureset = Array(Float64, length(features), ntrain_frames)
# targetset = Array(Float64, 2, ntrain_frames)
# interval_starts = Array(Int, nruns*ncars)

context = IntegratedContinuous(NGSIM_TIMESTEP,1)
models = Dict{Int, DriverModel}()
carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"blue"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"
carcolors[4] = colorant"green"
carcolors[5] = colorant"green"
carcolors[6] = colorant"green"
trajdata = Trajdata(roadway,
    Dict{Int, VehicleDef}(),
    Array(TrajdataState, nstates_total),
    Array(TrajdataFrame, nframes_per_run))

# extract the dataset for each segment
scene = Scene(ncars)

# Array containing speed values at each time step
speed_pass = zeros(6, 100)
speed_agg = zeros(6, 100)
speed_med1 = zeros(6, 100)
speed_med2 = zeros(6, 100)

d_agg = zeros(6, 100)
d_med2 = zeros(6, 100)

interval_index = 0
for run in 1:nruns
    println(run)

    if run <= 100
        scene = gen_start_state!(scene, context, models, passive=true)
    elseif run <= 200
        scene = gen_start_state!(scene, context, models, passive=false, aggressive=true)
    elseif run <= 300
        scene = gen_start_state!(scene, context, models, passive=false, medium1=true)
    else
        scene = gen_start_state!(scene, context, models, passive=false)
    end
    actions = get_actions!(Array(DriveAction, length(scene)), zeros(N_FEATURES), extractor, scene, roadway, models)

    # run sim and log to trajdata
    let
        i = 0
        t = 0.0
        frame_index = 0
        state_index = 0

        for veh in scene
            trajdata.vehdefs[veh.def.id] = veh.def
        end

        for frame in 1:nframes_per_run
            lo = state_index + 1
            for veh in scene
                trajdata.states[state_index+=1] = TrajdataState(veh.def.id, veh.state)
            end
            hi = state_index

            trajdata.frames[frame_index+=1] = TrajdataFrame(lo, hi, t)

            get_actions!(actions, zeros(N_FEATURES), extractor, scene, roadway, models)
            tick!(scene, roadway, actions, models)
            t += context.Δt
            i += 1

            # Store speed values
            if ((i % 10) == 0) || (i == 1)
                vehicle_index = get_index_of_first_vehicle_with_id(scene, 1)
                veh_ego = models[1].rec[1, 0]
                speed = veh_ego.state.v
                s_gap = get_frenet_relative_position(get_rear_center(scene[2]), veh_ego.state.posF.roadind, roadway).Δs
                d = max(0.0, s_gap)
                if run <= 100
                    if i == 1
                        speed_pass[1, run] = speed
                    else
                        speed_pass[Int(i/10)+1, run] = speed
                    end
                elseif run <= 200
                    if i == 1
                        speed_agg[1, run-100] = speed
                        d_agg[1, run-100] = d
                    else
                        speed_agg[Int(i/10)+1, run-100] = speed
                        d_agg[Int(i/10)+1, run-100] = d
                    end
                elseif run <= 300
                    if i == 1
                        speed_med1[1, run-200] = speed
                    else
                        speed_med1[Int(i/10)+1, run-200] = speed
                    end
                else
                    if i == 1
                        speed_med2[1, run-300] = speed
                        d_med2[1, run-300] = d
                    else
                        speed_med2[Int(i/10)+1, run-300] = speed
                        d_med2[Int(i/10)+1, run-300] = d
                    end
                end
            end
        end
        # frames = Frames(MIME("image/png"), fps=10)
        # for frame in 1 : nframes_per_run
        #     s = render(get!(scene, trajdata, frame), roadway, cam=CarFollowCamera(1, 10.0), car_colors = carcolors) 
        #     push!(frames, s)
        # end
        # println("creating gif...")
        # write("run_" * string(run) * ".gif", frames)
    end

    # open(io->write(io, trajdata), @sprintf("trajdata_passive_aggressive%d.txt", run), "w")
end
println("Saving to file...")
JLD.save("headways.jld", "speed_agg", speed_agg, "speed_pass", speed_pass, "speed_med1", speed_med1, "speed_med2", speed_med2, "d_med2", d_med2, "d_agg", d_agg)
