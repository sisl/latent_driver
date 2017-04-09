include("load_evaluation_data.jl")
include("Auto2D.jl")
include("load_networks.jl")
include("additional_metrics.jl")

const METRIC_SAVE_FILE_DIR = dirname(@__FILE__) * "/results/"

const N_SEGMENTS = 1000
const N_SIMULATIONS_PER_TRACE = 20
const EVAL_PRIME_DURATION = 5.0 # [s] - length of priming period
const EVAL_PRIME_STEPS = 50
const EVAL_DURATION = 5.0 # [s] - length of trace
const EVAL_DURATION_STEPS = 50
const CONTEXT = IntegratedContinuous(NGSIM_TIMESTEP, 3)
const METRICS = TraceMetricExtractor[
    RootWeightedSquareError(SPEED,  0.5),
    RootWeightedSquareError(SPEED,  1.0),
    RootWeightedSquareError(SPEED,  2.0),
    RootWeightedSquareError(SPEED,  3.0),
    RootWeightedSquareError(SPEED,  4.0),
    RootWeightedSquareError(SPEED,  5.0),

    RootWeightedSquareError(POSFS,  0.5),
    RootWeightedSquareError(POSFS,  1.0),
    RootWeightedSquareError(POSFS,  2.0),
    RootWeightedSquareError(POSFS,  3.0),
    RootWeightedSquareError(POSFS,  4.0),
    RootWeightedSquareError(POSFS,  5.0),

    EmergentKLDivergence(INV_TTC, 0., 10., 100),
    EmergentKLDivergence(SPEED, -5., 40., 100),
    EmergentKLDivergence(ACC, -5., 5., 100),
]

const EXTRACT_CORE = true
const EXTRACT_TEMPORAL = false
const EXTRACT_WELL_BEHAVED = true
const EXTRACT_NEIGHBOR_FEATURES = false
const EXTRACT_CARLIDAR_RANGERATE = true
const CARLIDAR_NBEAMS = 20
const ROADLIDAR_NBEAMS = 0
const ROADLIDAR_NLANES = 2
const CARLIDAR_MAX_RANGE = 100.0
const ROADLIDAR_MAX_RANGE = 100.0

println("Loading evaluation data...")
evaldata, classes = load_evaldata()
println("Done.")

# Check if trajectory occurs entirely on upper or lower straightaway
function on_straightaway(x_init, x_mid, x_final)
    if x_final > x_init
        if (10.0 <= x_init <= 240.0) && (10.0 <= x_final <= 220.0) && (x_mid < x_final)
            return true
        else
            return false
        end
    elseif x_final < x_init
        if (10.0 <= x_init <= 240.0) && (20.0 <= x_final <= 240.0) && (x_mid > x_final)
            return true
        else
            return false
        end
    else
        return false
    end
end

# Extract subset of validation data for evaluation
function create_evaldata(evaldata::EvaluationData, classes::Vector{Int}; nsegs::Int=1, nframes::Int=101)
    segments = Array(TrajdataSegment, nsegs)
    new_classes = Array(Int, nsegs)
    i = 0
    seg_exclude = [] # Array of trajectories to exclude
    while i < nsegs
        seg = rand(1:length(evaldata.segments))
        if !(seg in seg_exclude)
            # Make sure segment takes place exclusively on the straightaway
            segment = evaldata.segments[seg]
            trajdata = evaldata.trajdatas[segment.trajdata_index]
            rec = pull_record(segment, trajdata, 0)

            # Find initial, final, and intermediate positions along lane in Frenet frame
            vehicle_index = get_index_of_first_vehicle_with_id(rec, segment.egoid, 0)
            veh_ego = rec[vehicle_index, 0]
            x_final = veh_ego.state.posG.x

            vehicle_index = get_index_of_first_vehicle_with_id(rec, segment.egoid, -EVAL_DURATION_STEPS)
            veh_ego = rec[vehicle_index, -EVAL_DURATION_STEPS]
            x_init = veh_ego.state.posG.x

            vehicle_index = get_index_of_first_vehicle_with_id(rec, segment.egoid, -round(Int, EVAL_DURATION_STEPS/2))
            veh_ego = rec[vehicle_index, -round(Int, EVAL_DURATION_STEPS/2)]
            x_mid = veh_ego.state.posG.x

            if on_straightaway(x_init, x_mid, x_final)
                new_classes[i += 1] = classes[seg]
                segments[i] = segment
            end
            push!(seg_exclude, seg)
        end
    end
    EvaluationData(evaldata.trajdatas, segments), new_classes
end

function create_simparams(evaldata::EvaluationData, classes::Vector{Int})
    # Construct extractor
    extractor = Auto2D.MultiFeatureExtractor(
        EXTRACT_CORE,
        EXTRACT_TEMPORAL,
        EXTRACT_WELL_BEHAVED,
        EXTRACT_NEIGHBOR_FEATURES,
        EXTRACT_CARLIDAR_RANGERATE,
        CARLIDAR_NBEAMS,
        ROADLIDAR_NBEAMS,
        ROADLIDAR_NLANES,
        carlidar_max_range=CARLIDAR_MAX_RANGE,
        roadlidar_max_range=ROADLIDAR_MAX_RANGE,
        )

    # Convert array of trajdatas to dict
    trajdatas = Dict(zip(collect(1:length(evaldata.trajdatas)), evaldata.trajdatas))

    # Construct and return simparams
    Auto2D.SimParams(trajdatas, evaldata.segments, classes, 1, EVAL_PRIME_STEPS, EVAL_DURATION_STEPS, AccelTurnrate, extractor)
end

srand(0)
eval_seg_nframes = ceil(Int, (EVAL_PRIME_DURATION + EVAL_DURATION)/NGSIM_TIMESTEP) + 1
VALDATA_SUBSET, classes = create_evaldata(evaldata, classes, nsegs=N_SEGMENTS, nframes=eval_seg_nframes)
FOLDSET_TEST = foldset_match(fill(1, N_SEGMENTS), 1)
SIMPARAMS = create_simparams(VALDATA_SUBSET, classes)

function load_models(; context::IntegratedContinuous = CONTEXT)
    models = Dict()

    filepath = "./models/bc_fc.h5"
    models["bc_mlp"] = load_mlp_policy(filepath)

    filepath = "./models/bc_lstm.h5"
    models["bc_lstm"] = load_lstm(filepath)

    filepath = "./models/oracle_mlp.h5"
    models["oracle"] = load_mlp_policy(filepath, oracle=true)

    filepath = "./models/oracle_lstm.h5"
    models["oracle_lstm"] = load_lstm(filepath, oracle=true)

    filepath = "./models/policy_vae_lstm.h5"
    models["vae_lstm"] = load_lstm(filepath, use_latent=true)

    filepath = "./models/encoder_lstm.h5"
    models["encoder_lstm"] = load_lstm(filepath, encoder=true)
    

    models
end

# Validation for individual model
function validate(model::DriverModel;
    simparams::Auto2D.SimParams = SIMPARAMS,
    metrics::Vector{TraceMetricExtractor} = METRICS,
    foldset::FoldSet = FOLDSET_TEST,
    n_simulations_per_trace::Int = N_SIMULATIONS_PER_TRACE,
    save::Bool=true,
    modelname::AbstractString=AutomotiveDrivingModels.get_name(model)

    )

    metrics_df = allocate_metrics_dataframe(METRICS, 1)
    calc_metrics!(metrics_df, model, metrics, simparams, foldset,
                        n_simulations_per_trace=n_simulations_per_trace,
                        row = 1, prime_history=EVAL_PRIME_STEPS)

    if save
        filename = "valid_"*modelname*".csv"
        writetable(joinpath(METRIC_SAVE_FILE_DIR, filename), metrics_df)
    end

    metrics_df
end

# Validation for VAE
function validate(model1::DriverModel, 
    model2::Auto2D.LatentEncoder;
    simparams::Auto2D.SimParams = SIMPARAMS,
    metrics::Vector{TraceMetricExtractor} = METRICS,
    foldset::FoldSet = FOLDSET_TEST,
    n_simulations_per_trace::Int = N_SIMULATIONS_PER_TRACE,
    save::Bool=true,
    modelname::AbstractString=AutomotiveDrivingModels.get_name(model1)

    )

    metrics_df = allocate_metrics_dataframe(METRICS, 1)
    calc_metrics!(metrics_df, model1, model2, metrics, simparams, foldset,
                        n_simulations_per_trace=n_simulations_per_trace,
                        row = 1, prime_history=EVAL_PRIME_STEPS)

    if save
        filename = "valid_"*modelname*".csv"
        writetable(joinpath(METRIC_SAVE_FILE_DIR, filename), metrics_df)
    end

    metrics_df
end

"DONE"
