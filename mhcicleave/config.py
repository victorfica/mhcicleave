from importlib import resources

# Feature window parameters
UPSTREAM = 6
DOWNSTREAM = 4
SEQ_LEN = UPSTREAM + DOWNSTREAM

# ESM-2 embedding dimensions
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_DIM = 1280

# Default MLP Architecture
DEFAULT_MLP_PARAMS = {
    "input_size": SEQ_LEN * ESM2_DIM,
    "hidden_linear1": 256,
    "hidden_linear2": 128,
    "out_neurons": 1,
    "dropout": 0.5
}


# Dynamically resolve the path to the bundled checkpoint
try:
    # This looks inside the 'mhcicleave.weights' module and grabs the file path
    DEFAULT_CHECKPOINT = str(
        resources.files('mhcicleave.models').joinpath('best_model-epoch=13-val_loss=0.32.ckpt')
    )
except Exception as e:
    # Fallback/warning just in case the file wasn't packaged correctly
    print(f"WARNING: Bundled checkpoint not found. Ensure 'best_model.ckpt' is in 'mhcicleave/models/'. Error: {e}")
    DEFAULT_CHECKPOINT = None