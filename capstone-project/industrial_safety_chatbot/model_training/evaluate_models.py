# evaluate_models.py
import json

with open("metrics/ml_metrics.json") as f:
    ml = json.load(f)

with open("metrics/lstm_metrics.json") as f:
    lstm = json.load(f)

with open("metrics/fnn_metrics.json") as f:
    fnn = json.load(f)

with open("metrics/bilstm_metrics.json") as f:
    bilstm = json.load(f)

all_models = {}

# Merge ML models
for name, data in ml.items():
    all_models[name] = {
        "accuracy": data["accuracy"],
        "f1_score": data["macro avg"]["f1-score"],
    }

# Add LSTM
all_models["LSTM"] = {
    "accuracy": lstm["accuracy"],
    "f1_score": lstm["macro avg"]["f1-score"],
}

all_models["FNN"] = {
    "accuracy": fnn["accuracy"],
    "f1_score": fnn["macro avg"]["f1-score"],
}

all_models["BiLSTM-GloVe"] = {
    "accuracy": bilstm["accuracy"],
    "f1_score": bilstm["macro avg"]["f1-score"],
}

# Find best model by F1
best_model = max(all_models.items(), key=lambda x: x[1]["f1_score"])

# Save comparison
with open("metrics/comparison.json", "w") as f:
    json.dump(
        {
            "all_models": all_models,
            "best_model": {"name": best_model[0], "metrics": best_model[1]},
        },
        f,
        indent=2,
    )

print(f"✅ Best model: {best_model[0]} with F1: {best_model[1]['f1_score']:.4f}")
