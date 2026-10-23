import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Must match train_model.py exactly. No attack-name inputs are accepted.
NUMERIC_FEATURES = [
    "count",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate",
    "dst_host_srv_count",
    "last_flag",
    "logged_in",
    "same_srv_rate",
    "serror_rate",
]
CATEGORICAL_FEATURES = ["flag", "service"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASS_LABELS = {
    "normal": "Normal traffic",
    "dos": "DoS (Denial of Service)",
    "probe": "Probe / scanning",
    "r2l": "R2L (Remote-to-Local)",
    "u2r": "U2R (User-to-Root)",
}

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

def build_input(data):
    cleaned = {}

    for name in NUMERIC_FEATURES:
        value = data.get(name)
        if value is None or str(value).strip() == "":
            raise ValueError(f"Please enter {name.replace('_', ' ')}.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name.replace('_', ' ')} must be numeric.")
        if not np.isfinite(number):
            raise ValueError(f"{name.replace('_', ' ')} must be a finite number.")
        cleaned[name] = number

    for name in (
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_same_srv_rate",
        "same_srv_rate",
        "serror_rate",
    ):
        if not 0 <= cleaned[name] <= 1:
            raise ValueError(f"{name.replace('_', ' ')} must be between 0 and 1.")

    for name in ("count", "dst_host_srv_count", "last_flag"):
        if cleaned[name] < 0 or not cleaned[name].is_integer():
            raise ValueError(f"{name.replace('_', ' ')} must be a non-negative whole number.")
        cleaned[name] = int(cleaned[name])

    if cleaned["logged_in"] not in (0, 1):
        raise ValueError("Login status must be 0 (No) or 1 (Yes).")
    cleaned["logged_in"] = int(cleaned["logged_in"])

    for name in CATEGORICAL_FEATURES:
        value = str(data.get(name, "")).strip().lower()
        if not value:
            raise ValueError(f"Please select {name}.")
        cleaned[name] = value

    return pd.DataFrame([{name: cleaned[name] for name in FEATURES}], columns=FEATURES)

def predict_result(data):
    if model is None:
        raise RuntimeError(
            "The retrained model is missing. Put the dataset in NSL_Dataset/ and run "
            "`python train_model.py` before starting the app."
        )
    row = build_input(data)
    predicted = str(model.predict(row)[0]).lower()
    score = None
    if hasattr(model, "predict_proba"):
        score = float(np.max(model.predict_proba(row)[0]))
    return {
        "prediction": predicted,
        "label": CLASS_LABELS.get(predicted, predicted.upper()),
        "confidence_estimate": score,
        "warning": "Benchmark-model estimate only; this does not prove an attack occurred.",
    }

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    if request.method == "POST":
        try:
            result = predict_result(request.form)
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Confirm model.pkl was created by the matching train_model.py."
    return render_template("index.html", result=result, error=error)

@app.route("/results", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def results():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object containing the named form fields."}), 400
    try:
        return jsonify(predict_result(data))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("API prediction failed")
        return jsonify({"error": "Prediction failed. Confirm the model matches the current feature schema."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="127.0.0.1", port=port, debug=False)
