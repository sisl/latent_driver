using AutomotiveDrivingModels
using NGSIM
using HDF5
using JLD


const TRAJDATA_FILEPATH = "/Users/jeremymorton/Documents/sisl/summer_2016/irl/pull_traces/"
const NUM_TRAJDATA = 30     # Number of trajdatas in validation set
const NUM_VEHICLES = 33     # Number of vehicles in each scene
const NUM_SEG = 3           # Number of segments to divide each trajdata into

export
    load_evaldata

# Load trajdata from given file
function load_trajdata(i::Int)
    filepath = TRAJDATA_FILEPATH * "trajdata_passive_aggressive" * string(i) * ".txt"
    td = open(io->read(io, Trajdata), filepath, "r")
    td.roadway = gen_stadium_roadway(3, length=250.0, radius=45.0)
    td
end

# Load dictionaries mapping vehicle to driving class
function load_classdict(i::Int)
    filepath = TRAJDATA_FILEPATH * "car_classes_" * string(i) * ".jld"
    cd = JLD.load(filepath, "classes")
    cd
end

function load_evaldata()
    # Extract trajdatas
    trajdatas = map(i->load_trajdata(i), 1:NUM_TRAJDATA)
    classdicts = map(i->load_classdict(i), 1:NUM_TRAJDATA)

    # Loop over each trajdata, vehicle, and segment
    segments = Array(TrajdataSegment, NUM_TRAJDATA*NUM_VEHICLES*NUM_SEG)
    classes = Array(Int, NUM_TRAJDATA*NUM_VEHICLES*NUM_SEG)
    for i = 1:NUM_TRAJDATA
        for j = 1:NUM_VEHICLES
            for k = 1:NUM_SEG
                # Find index for individual segment
                idx = NUM_VEHICLES*NUM_SEG*(i-1) + NUM_SEG*(j-1) + k
                segments[idx] = TrajdataSegment(i, j, (k-1)*100+1, k*100)
                classes[idx] = classdicts[i][j]     # dict of dicts, need to index twice
            end
        end
    end

    EvaluationData(trajdatas, segments), classes
end