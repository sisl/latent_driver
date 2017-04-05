include("validation.jl")

# Load models
models = load_models()

# Run validation for each model
println("Running validation for VAE...")
validate(models["vae_new"], models["encoder_new"], modelname="vae_long")

println("Running validation for MLP...")
validate(models["bc_mlp"], modelname="mlp_long")

println("Running validation for LSTM...")
validate(models["bc_lstm"], modelname="lstm_long")

println("Running validation for Oracle...")
validate(models["oracle"], modelname="oracle_long")

println("Running validation for Oracle LSTM...")
validate(models["oracle_lstm"], modelname="oracle_lstm")

# println("Running validation for VAE 1-d z...")
# validate(models["vae_1"], models["encoder_1"], modelname="vae_1")

# println("Running validation for VAE 3-d z...")
# validate(models["vae_3"], models["encoder_3"], modelname="vae_3")



