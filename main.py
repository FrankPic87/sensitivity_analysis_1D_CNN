from pathlib import Path
import time
import joblib
import numpy as np
import torch
import torch.nn as nn

# Import execution modules from dedicated scripts
from linearization import run_linearization
from mc_classic import run_mc_classic
from mc_dropout import run_mc_dropout

# ============================================================
# GENERAL CONFIGURATION
# ============================================================
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
WEIGHTS_PATH = SCRIPT_DIRECTORY / "model_weights.pt"
SCALER_PATH = SCRIPT_DIRECTORY / "scaler.pkl"

QUERY_VALUES = [19.07, 19.50, 91.40, 615.0]  # Raw physical values
NUMBER_ITERATIONS = 50000
CLASSIFICATION_THRESHOLD = 0.5
GAUSSIAN_NOISE_SIGMA = 0.1  # Noise standard deviation in normalized space
INPUT_CLIP_RANGE = None
RANDOM_SEED = 42


# ============================================================
# CNN MODEL DEFINITION
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_length, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        conv_out = input_length - 2 + 1
        self.fc = nn.Linear(4 * conv_out, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return torch.sigmoid(x)


# ============================================================
# PARAMETER LOADING AND UTILITIES
# ============================================================
def load_model_state_dict(weights_path):
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Invalid weight file format.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): v for key, v in state_dict.items()}

    return state_dict


def infer_input_length(state_dict):
    conv_weight = state_dict["conv.weight"]
    fc_weight = state_dict["fc.weight"]
    number_filters = conv_weight.shape[0]
    kernel_size = conv_weight.shape[-1]
    fc_input_features = fc_weight.shape[1]
    convolution_output_length = fc_input_features // number_filters
    return convolution_output_length + kernel_size - 1


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File existence checks
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights file not found at: {WEIGHTS_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")

    # Load fitted StandardScaler
    scaler = joblib.load(SCALER_PATH)

    # Load model weights and infer dimension
    state_dict = load_model_state_dict(WEIGHTS_PATH)
    input_length = infer_input_length(state_dict)

    if len(QUERY_VALUES) != input_length:
        raise ValueError(f"Query sequence must contain exactly {input_length} elements.")

    # ------------------------------------------------------------
    # SCALE RAW QUERY VALUES
    # Convert 1D list to 2D numpy array (1, n_features) before scaling
    # ------------------------------------------------------------
    raw_query_array = np.array([QUERY_VALUES])
    scaled_query = scaler.transform(raw_query_array)

    # Convert scaled values to PyTorch tensor with shape (1, 1, input_length)
    query = torch.tensor(scaled_query, dtype=torch.float32, device=device).reshape(1, 1, input_length)

    # Model initialization and weight assignment
    model = SimpleCNN(input_length=input_length)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.requires_grad_(False)

    print("=" * 70)
    print("UNCERTAINTY PROPAGATION BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Raw Query Input             : {QUERY_VALUES}")
    print(f"Scaled Query Tensor         : {query.squeeze().tolist()}")
    print(f"Simulation iterations (K)   : {NUMBER_ITERATIONS}")
    print(f"Input noise standard dev.   : {GAUSSIAN_NOISE_SIGMA}")
    print("-" * 70)

    # 1. CONVENTIONAL MONTE CARLO SIMULATION
    t0 = time.perf_counter()
    mc_classic_res = run_mc_classic(
        model=model,
        query=query,
        noise_sigma=GAUSSIAN_NOISE_SIGMA,
        number_iterations=NUMBER_ITERATIONS,
        threshold=CLASSIFICATION_THRESHOLD,
        clip_range=INPUT_CLIP_RANGE,
    )
    time_mc_classic = time.perf_counter() - t0

    # 2. MONTE CARLO DROPOUT (Optimized: Dropout layer repetition only)
    t0 = time.perf_counter()
    mc_dropout_res = run_mc_dropout(
        model=model,
        query=query,
        noise_sigma=GAUSSIAN_NOISE_SIGMA,
        number_iterations=NUMBER_ITERATIONS,
        threshold=CLASSIFICATION_THRESHOLD,
    )
    time_mc_dropout = time.perf_counter() - t0

    # 3. LINEARIZATION-BASED TECHNIQUE (Analytical)
    t0 = time.perf_counter()
    linear_res = run_linearization(
        model=model,
        query=query,
        noise_sigma=GAUSSIAN_NOISE_SIGMA,
    )
    time_linearization = time.perf_counter() - t0

    # ============================================================
    # RESULTS REPORT AND COMPUTATIONAL TIMINGS
    # ============================================================
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS & COMPUTATIONAL PERFORMANCE")
    print("=" * 70)
    print(f"{'Method':<25} | {'Time [s]':<12} | {'P(Class 0)':<12} | {'P(Class 1)':<12} | {'E[Sigmoid]':<12}")
    print("-" * 70)

    print(
        f"{'Conventional MC':<25} | {time_mc_classic:<12.5f} | "
        f"{mc_classic_res['relative_frequency_class_0']:<12.4f} | "
        f"{mc_classic_res['relative_frequency_class_1']:<12.4f} | "
        f"{mc_classic_res['mean_sigmoid']:<12.4f}"
    )

    print(
        f"{'MC Dropout (Optimized)':<25} | {time_mc_dropout:<12.5f} | "
        f"{mc_dropout_res['relative_frequency_class_0']:<12.4f} | "
        f"{mc_dropout_res['relative_frequency_class_1']:<12.4f} | "
        f"{mc_dropout_res['mean_sigmoid']:<12.4f}"
    )

    print(
        f"{'Linearized (Analytical)':<25} | {time_linearization:<12.5f} | "
        f"{linear_res['relative_frequency_class_0']:<12.4f} | "
        f"{linear_res['relative_frequency_class_1']:<12.4f} | "
        f"{linear_res['expected_sigmoid']:<12.4f}"
    )
    print("=" * 70)

    # Speedup calculation relative to Conventional MC
    speedup_dropout = time_mc_classic / time_mc_dropout if time_mc_dropout > 0 else 0
    speedup_linear = time_mc_classic / time_linearization if time_linearization > 0 else 0

    print("\n--- PERFORMANCE & SPEEDUP SUMMARY ---")
    print(f"Speedup MC Dropout vs Conventional MC   : {speedup_dropout:.2f}x")
    print(f"Speedup Linearization vs Conventional MC: {speedup_linear:.2f}x")


if __name__ == "__main__":
    main()