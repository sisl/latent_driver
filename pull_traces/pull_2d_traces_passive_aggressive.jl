using AutomotiveDrivingModels
using NGSIM
using HDF5, JLD
using AutoViz
using Reel
using PGFPlots

include("multifeatureset.jl")
# include("tim2dfeatures.jl")
include("passive_aggressive.jl")

# srand(0)

# generate roadway
roadway = PASSIVE_AGGRESSIVE_ROADWAY
open(io->write(io, roadway), "roadway_passive_aggressive.txt", "w")

# starting locations
start_roadinds = START_ROADINDS
start_delta_s = START_DELTA_S
ncars = length(start_roadinds) * length(start_delta_s)
nframes_per_run = ceil(Int, 15.0 / NGSIM_TIMESTEP) # 30 seconds of driving
nruns = 30
nstates_total = ncars*nruns*nframes_per_run
ntrain_frames = ncars*nruns*(nframes_per_run-1)

println("ncars: ", ncars)
println("nframes_per_run: ", nframes_per_run)
println("nruns: ", nruns)
println("nstates_total: ", nstates_total)
println("ntrain_frames: ", ntrain_frames)

# extract the dataset for each segment
scene = Scene(ncars)
rec = SceneRecord(2, NGSIM_TIMESTEP, ncars)

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
featureset = Array(Float64, length(features), ntrain_frames)
targetset = Array(Float64, 2, ntrain_frames)
class_set = Array(Float64, ntrain_frames)
interval_starts = Array(Int, nruns*ncars)

context = IntegratedContinuous(NGSIM_TIMESTEP,1)
models = Dict{Int, DriverModel}()
trajdata = Trajdata(roadway,
    Dict{Int, VehicleDef}(),
    Array(TrajdataState, nstates_total),
    Array(TrajdataFrame, nframes_per_run))

i = 0
interval_index = 0
for run in 1:nruns
    println(run)

    scene, carcolors, car_class = gen_start_state!(scene, context, models)
    # carcolors[1] = colorant"white"
    actions = get_actions!(Array(DriveAction, length(scene)), scene, roadway, models)

    # run sim and log to trajdata
    let
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

            get_actions!(actions, scene, roadway, models)
            tick!(scene, roadway, actions, models)
            t += context.Δt
        end
        # frames = Frames(MIME("image/png"), fps=10)
        # for frame in 1 : nframes_per_run
        #     s = render(get!(scene, trajdata, frame), roadway, cam=FitToContentCamera(0.1), car_colors = carcolors) 
        #     push!(frames, s)
        # end
        # println("creating gif...")
        # write("passive_aggressive.gif", frames)
    end

    open(io->write(io, trajdata), @sprintf("trajdata_passive_aggressive%d.txt", run), "w")
    JLD.save("car_classes_" * string(run) * ".jld", "classes", car_class)

    # extract features
    for id in keys(trajdata.vehdefs)
        empty!(rec)
        interval_starts[interval_index+=1] = i+1
        for frame in 1 : nframes_per_run
            get!(scene, trajdata, frame)
            update!(rec, scene)

            vehicle_index = get_index_of_first_vehicle_with_id(rec, id)
            pull_features!(extractor, features, rec, trajdata.roadway, vehicle_index)

            if frame > 1
                targetset[1, i] = get(ACC, rec, trajdata.roadway, vehicle_index)
                targetset[2, i] = get(TURNRATEG, rec, trajdata.roadway, vehicle_index)
                class_set[i] = 3#car_class[id]
            end

            if frame < nframes_per_run
                i += 1
                featureset[:,i] = features
            end
        end
    end
end

@assert(length(interval_starts) == interval_index)
@assert(i == ntrain_frames)

# Save it all to a file
outfile = "../2d_drive_data/data_distinct_drivers_4.jld"
JLD.save(outfile, "features", featureset, "targets", targetset, "intervals", interval_starts, "timestep", NGSIM_TIMESTEP, "classes", class_set)

