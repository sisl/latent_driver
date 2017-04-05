using AutomotiveDrivingModels
using AutoViz
using Reel
using PGFPlots

########################################
#        RWSE Global Position          #
########################################

type RWSEPosG <: TraceMetricExtractor
    horizon::Float64 # [s]
    running_sum::Float64
    n_obs::Int
end
RWSEPosG(horizon::Float64) = RWSEPosG(horizon, 0.0, 0)

Base.Symbol(m::RWSEPosG) = Symbol(@sprintf("RWSE_posG_%d_%02d", floor(Int, m.horizon), floor(Int, 100*rem(m.horizon, 1.0))))
get_score(m::RWSEPosG) = sqrt(m.running_sum / m.n_obs)
function reset!(metric::RWSEPosG)
    metric.running_sum = 0.0
    metric.n_obs = 0
    metric
end
function extract!(
    metric::RWSEPosG,
    rec_orig::SceneRecord, # the records are exactly as long as the simulation (ie, contain everything)
    rec_sim::SceneRecord,
    roadway::Roadway,
    egoid::Int,
    prime_history::Int,
    )

    # TODO: how to handle missing values???

    pastframe = 1-length(rec_orig) + clamp(round(Int, metric.horizon/rec_orig.timestep) + prime_history, 0, length(rec_orig)-1)

    # pull true value
    vehicle_index = get_index_of_first_vehicle_with_id(rec_orig, egoid, pastframe)
    veh_ego = rec_orig[vehicle_index, pastframe]
    x_true = veh_ego.state.posG.x
    y_true = veh_ego.state.posG.y

    # pull sim value
    vehicle_index = get_index_of_first_vehicle_with_id(rec_sim, egoid, pastframe)
    veh_ego = rec_sim[vehicle_index, pastframe]
    x_montecarlo = veh_ego.state.posG.x
    y_montecarlo = veh_ego.state.posG.y

    d = (x_true - x_montecarlo)^2 + (y_true - y_montecarlo)^2
    
    metric.running_sum += d
    metric.n_obs += 1

    metric
end

function rollout!(
    rec::SceneRecord,
    model::DriverModel,
    encoder::Auto2D.LatentEncoder,
    egoid::Int,
    trajdata::Trajdata,
    time_start::Float64,
    time_end::Float64,
    simparams::Auto2D.SimParams;
    prime_history::Int = 0,
    )
    
    # Initialize values
    Δt = rec.timestep
    simstate = simparams.simstates[1]

    # Reinitialize internal state of encoder and clear record
    zero!(encoder.net[:LSTM_0])
    zero!(encoder.net[:LSTM_1])
    # Reinitialize internal sate of policy if recurrent
    if :LSTM_0 in keys(model.net.name_to_index)
        zero!(model.net[:LSTM_0])
        zero!(model.net[:LSTM_1])
    end
    empty!(simstate.rec)
    empty!(rec)
    reset_hidden_state!(model)

    # Prime model (will do nothing for mlp)
    t = time_start + prime_history*Δt
    while t < time_end
        update!(simstate.rec, get!(simstate.scene, trajdata, t))
        observe!(encoder, simparams, simstate.scene, trajdata.roadway, simstate.egoid)
        t += Δt
    end
    # Create sample for latent state
    Auto2D.set_latent_state!(model, rand(encoder))

    # Reinitialize everything to end of priming period
    t = time_start + prime_history*Δt
    simstate.frame = prime_history
    update!(simstate.rec, get!(simstate.scene, trajdata, t))
    observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)

    # run simulation
    while t < time_end
        # Find action and step forward
        ego_action = rand(model)
        a = clamp(ego_action.a, -5.0, 5.0)
        ω = 0.0 #clamp(ego_action.ω, -0.1, 0.1)
        Auto2D.step(simparams, [a, ω])

        # update record
        observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)
        update!(rec, simstate.scene)

        # update time
        t += Δt
    end
    rec
end

function rollout!(
    rec::SceneRecord,
    model::DriverModel,
    egoid::Int,
    trajdata::Trajdata,
    time_start::Float64,
    time_end::Float64,
    simparams::Auto2D.SimParams;
    prime_history::Int = 0,
    )
    
    # Initialize values
    Δt = rec.timestep
    simstate = simparams.simstates[1]

    # Reinitialize internal sate and clear record
    if :LSTM_0 in keys(model.net.name_to_index)
        zero!(model.net[:LSTM_0])
        zero!(model.net[:LSTM_1])
    end
    # clear rec and make first observations
    if :GAIL_0 in keys(model.net.name_to_index)
        zero!(model.net[:gru])
    end
    empty!(simstate.rec)
    empty!(rec)
    reset_hidden_state!(model)

    # Prime model (will do nothing for mlp)
    t = time_start
    for i in 1 : prime_history
        simstate.frame += 1
        update!(simstate.rec, get!(simstate.scene, trajdata, t))
        observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)
        t += Δt
    end
    # run simulation
    while t < time_end

        # Find action and step forward
        ego_action = rand(model)
        a = clamp(ego_action.a, -5.0, 5.0)
        ω = 0.0 #clamp(ego_action.ω, -0.005, 0.005)
        Auto2D.step(simparams, [a, ω])

        # update record
        observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)
        update!(rec, simstate.scene)

        # update time
        t += Δt
    end
    rec
end

# Initialize values for simstate
function reset_simstate!(simstate::Auto2D.SimState, seg::TrajdataSegment)
    simstate.egoid = seg.egoid
    simstate.trajdata_index = seg.trajdata_index
    simstate.frame = seg.frame_lo
end

########################################
#        Calculate Metric Scores       #
########################################

function calc_metrics!(
    metrics_df::DataFrame,
    model::DriverModel,
    metrics::Vector{TraceMetricExtractor},
    simparams::Auto2D.SimParams,
    foldset_seg_test::FoldSet; # should match the test segments in evaldata
    n_simulations_per_trace::Int = 10,
    row::Int = foldset_seg_test.fold, # row in metrics_df to write to
    prime_history::Int = 0,
    calc_logl::Bool = true,
    )
    # reset metrics
    for metric in metrics
        try
            reset!(metric)
        catch
            AutomotiveDrivingModels.reset!(metric)
        end
    end
    n_traces = 0

    # simulate traces and perform online metric extraction
    scene = Scene()
    for seg_index in foldset_seg_test
        seg = simparams.segments[seg_index]
        trajdata = simparams.trajdatas[seg.trajdata_index]

        # Set driver class to appropriate value
        # println(simparams.classes[seg_index])
        Auto2D.set_driver_class!(model, simparams.classes[seg_index])

        # Initialize
        rec_orig = pull_record(seg, trajdata, 0) # TODO - make efficient
        rec_sim = deepcopy(rec_orig)

        # frames = Frames(MIME("image/png"), fps=10)
        # for pastframe in -50 : 0
        #     s = render(get_scene(rec_orig, pastframe), trajdata.roadway, [CarFollowingStatsOverlay(1, 2)], cam=CarFollowCamera(seg.egoid, 4.0), 
        #        car_colors=Dict{Int,Colorant}(seg.egoid=>COLOR_CAR_EGO)) 
        #     push!(frames, s)
        # end
        # println("creating gif...")
        # write("video_orig" * string(simparams.classes[seg_index]) * ".gif", frames)

        time_start = get_time(trajdata, seg.frame_lo)
        time_end = get_time(trajdata, seg.frame_hi)

        n_traces += 1

        for sim_index in 1 : n_simulations_per_trace
            reset_simstate!(simparams.simstates[1], seg)
            rollout!(rec_sim, model, seg.egoid, trajdata,
                      time_start, time_end, simparams, prime_history=prime_history)      

            for metric in metrics
                try
                    extract!(metric, rec_orig, rec_sim, trajdata.roadway, seg.egoid, prime_history)
                catch
                    AutomotiveDrivingModels.extract!(metric, rec_orig, rec_sim, trajdata.roadway, seg.egoid, prime_history)
                end
            end
        end
    end

    # compute metric scores
    for metric in metrics
        try
            metrics_df[row, Symbol(metric)] = get_score(metric)
        catch
            metrics_df[row, Symbol(metric)] = AutomotiveDrivingModels.get_score(metric)
        end
    end
    metrics_df[row, :time] = string(now())

    metrics_df
end

function calc_metrics!(
    metrics_df::DataFrame,
    model::DriverModel,
    encoder::Auto2D.LatentEncoder,
    metrics::Vector{TraceMetricExtractor},
    simparams::Auto2D.SimParams,
    foldset_seg_test::FoldSet; # should match the test segments in evaldata
    n_simulations_per_trace::Int = 10,
    row::Int = foldset_seg_test.fold, # row in metrics_df to write to
    prime_history::Int = 0,
    calc_logl::Bool = true,
    )
    # reset metrics
    for metric in metrics
        try
            reset!(metric)
        catch
            AutomotiveDrivingModels.reset!(metric)
        end
    end
    n_traces = 0

    # simulate traces and perform online metric extraction
    scene = Scene()
    for seg_index in foldset_seg_test
        seg = simparams.segments[seg_index]
        trajdata = simparams.trajdatas[seg.trajdata_index]

        rec_orig = pull_record(seg, trajdata, 0) # TODO - make efficient
        rec_sim = deepcopy(rec_orig)

        time_start = get_time(trajdata, seg.frame_lo)
        time_end = get_time(trajdata, seg.frame_hi)

        n_traces += 1

        for sim_index in 1 : n_simulations_per_trace
            reset_simstate!(simparams.simstates[1], seg)
            rollout!(rec_sim, model, encoder, seg.egoid, trajdata,
                      time_start, time_end, simparams, prime_history=prime_history)      

            for metric in metrics
                try
                    extract!(metric, rec_orig, rec_sim, trajdata.roadway, seg.egoid, prime_history)
                catch
                    AutomotiveDrivingModels.extract!(metric, rec_orig, rec_sim, trajdata.roadway, seg.egoid, prime_history)
                end
            end
        end
    end

    # compute metric scores
    for metric in metrics
        try
            metrics_df[row, Symbol(metric)] = get_score(metric)
        catch
            metrics_df[row, Symbol(metric)] = AutomotiveDrivingModels.get_score(metric)
        end
    end
    metrics_df[row, :time] = string(now())

    metrics_df
end
