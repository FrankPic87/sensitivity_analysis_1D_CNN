import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# =========================================================
# FEATURE UNITS DICTIONARY MAPPING
# =========================================================
BASE_UNITS = {
    "radius": "µm",
    "texture": "gray levels",
    "perimeter": "µm",
    "area": "µm²",
    "smoothness": "dimensionless",
    "compactness": "dimensionless",
    "concavity": "dimensionless",
    "concave points": "dimensionless",
    "symmetry": "dimensionless",
    "fractal dimension": "dimensionless",
}


def get_feature_unit(feature_name):
    """
    Returns the measurement unit associated with a given feature name.
    """
    for base_feature, unit in BASE_UNITS.items():
        if base_feature in feature_name:
            return unit
    return "N/A"


# =========================================================
# 1. CNN MODEL
# =========================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_length):
        super(SimpleCNN, self).__init__()

        # Single convolution layer
        self.conv = nn.Conv1d(1, 4, kernel_size=2)

        # Activation
        self.relu = nn.ReLU()

        # Compute flattened size
        conv_out = input_length - 2 + 1

        # Single fully connected layer
        self.fc = nn.Linear(4 * conv_out, 1)

        # Output activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.sigmoid(x)

        return x


# =========================================================
# 2. TRAINING FUNCTION
# =========================================================
def train_model(model, X_train, y_train, epochs=50):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")


# =========================================================
# 3. EVALUATION FUNCTION
# =========================================================
def evaluate(model, X, y, name="Set"):

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        probs = model(X_tensor).numpy().flatten()

    preds = (probs > 0.5).astype(int)

    cm = confusion_matrix(y, preds)
    acc = accuracy_score(y, preds)

    print(f"\n{name} Results")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return acc, cm


# =========================================================
# 4. MAIN PROGRAM
# =========================================================
if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)

    # -----------------------------
    # Load dataset
    # -----------------------------
    data = load_breast_cancer()
    all_feature_names = data.feature_names.tolist()

    print("=" * 85)
    print("FULL LIST OF AVAILABLE FEATURES WITH UNITS (30):")
    print("=" * 85)
    for idx, f_name in enumerate(all_feature_names):
        unit = get_feature_unit(f_name)
        print(f"[{idx:2d}] {f_name:<28} | Unit: {unit}")
    print("=" * 85)

    # -----------------------------
    # Manual feature selection
    # Specify desired features here
    # -----------------------------
    selected_feature_names = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area"
    ]

    # Verify that selected features exist
    invalid_features = [f for f in selected_feature_names if f not in all_feature_names]
    if invalid_features:
        raise ValueError(f"The following features do not exist in dataset: {invalid_features}")

    # =========================================================
    # FEATURE RANGE ANALYSIS (WITH MEASUREMENT UNITS)
    # =========================================================
    print("\n" + "=" * 85)
    print("FEATURE RANGE ANALYSIS ON RAW DATASET:")
    print("=" * 85)
    print(f"{'Feature Name':<25} | {'Unit':<15} | {'Min':<10} | {'Max':<10} | {'Range (Max-Min)':<15}")
    print("-" * 85)

    for f_name in selected_feature_names:
        idx = all_feature_names.index(f_name)
        raw_values = data.data[:, idx]

        unit = get_feature_unit(f_name)
        min_val = np.min(raw_values)
        max_val = np.max(raw_values)
        val_range = max_val - min_val

        print(f"{f_name:<25} | {unit:<15} | {min_val:<10.4f} | {max_val:<10.4f} | {val_range:<15.4f}")

    print("=" * 85)

    # -----------------------------
    # Extract selected feature columns
    # -----------------------------
    feature_indices = [all_feature_names.index(f) for f in selected_feature_names]
    X = data.data[:, feature_indices]
    y = data.target.astype(np.float32)

    # -----------------------------
    # Normalization
    # -----------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -----------------------------
    # Train / Validation / Test split
    # -----------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    # -----------------------------
    # Model initialization
    # -----------------------------
    input_len = len(selected_feature_names)
    model = SimpleCNN(input_length=input_len)

    # -----------------------------
    # Training
    # -----------------------------
    print(f"\nTraining model with input_length = {input_len}...")
    train_model(model, X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    evaluate(model, X_val, y_val, "Validation")
    evaluate(model, X_test, y_test, "Test")

    # =====================================================
    # 5. SAVE MODEL WEIGHTS
    # =====================================================
    torch.save(model.state_dict(), "model_weights.pt")
    joblib.dump(scaler, "scaler.pkl")

    print("\nModel weights saved to 'model_weights.pt'.")
    print("Scaler saved to 'scaler.pkl'.")