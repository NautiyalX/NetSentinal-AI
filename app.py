
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"

# Must match train_model.py exactly.
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

SEVERITY_LEVELS = {
    "normal": "Low",
    "probe": "Medium",
    "dos": "High",
    "r2l": "High",
    "u2r": "Critical",
}

# Common NSL-KDD connection flags.
ALLOWED_FLAGS = {
    "sf", "s0", "rej", "rstr", "s1", "s2", "s3",
    "sh", "rsto", "s3", "oth", "rstos0", "rsts0",
    "s2", "s0", "oth", "sh", "rsto", "rsto",
}

# Values accepted by the NSL-KDD service feature.
# Unknown categories are also supported by the model's OneHotEncoder,
# so the service field is checked for non-empty text rather than blocked.


def load_model():
    if not MODEL_PATH.is_file():
        app.logger.warning("Model file not found: %s", MODEL_PATH)
        return None

    loaded = joblib.load(MODEL_PATH)

    # A retrained model must be a pipeline with the expected input schema.
    if hasattr(loaded, "feature_names_in_"):
        model_features = list(loaded.feature_names_in_)
        if model_features != FEATURES:
            raise ValueError(
                "model.pkl feature schema does not match app.py. "
                f"Expected {FEATURES}, got {model_features}."
            )

    return loaded


try:
    model = load_model()
except Exception:
    app.logger.exception("Could not load model.pkl")
    model = None


def build_input(data):
    cleaned = {}

    # Validate numeric fields.
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

    # Rates must be between 0 and 1.
    rate_features = [
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_same_srv_rate",
        "same_srv_rate",
        "serror_rate",
    ]

    for name in rate_features:
        if not 0 <= cleaned[name] <= 1:
            raise ValueError(
                f"{name.replace('_', ' ')} must be between 0 and 1."
            )

    # Count fields must be non-negative integers.
    for name in ("count", "dst_host_srv_count", "last_flag"):
        if cleaned[name] < 0 or not cleaned[name].is_integer():
            raise ValueError(
                f"{name.replace('_', ' ')} must be a non-negative whole number."
            )
        cleaned[name] = int(cleaned[name])

    # Login status must be 0 or 1.
    if cleaned["logged_in"] not in (0, 1):
        raise ValueError("Login status must be 0 (No) or 1 (Yes).")

    cleaned["logged_in"] = int(cleaned["logged_in"])

    # Validate categorical fields.
    for name in CATEGORICAL_FEATURES:
        value = str(data.get(name, "")).strip().lower()

        if not value:
            raise ValueError(f"Please select {name}.")

        if name == "flag" and value not in ALLOWED_FLAGS:
            raise ValueError("Please select a valid NSL-KDD connection flag.")

        cleaned[name] = value

    # Keep column order identical to the training script.
    return pd.DataFrame(
        [{name: cleaned[name] for name in FEATURES}],
        columns=FEATURES,
    )


def save_prediction(prediction, confidence, severity):
    """Append a successful prediction to CSV history."""
    record = pd.DataFrame([{
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "predicted_class": prediction,
        "confidence_percent": (
            round(confidence * 100, 2)
            if confidence is not None
            else None
        ),
        "severity": severity,
    }])

    record.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not HISTORY_PATH.exists(),
        index=False,
    )


def predict_result(data):
    if model is None:
        raise RuntimeError(
            "The model is unavailable. Check model.pkl and retrain_model.py "
            "compatibility before predicting."
        )

    row = build_input(data)

    predicted = str(model.predict(row)[0]).lower()

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        confidence = float(np.max(probabilities))

    severity = SEVERITY_LEVELS.get(predicted, "Unknown")
    save_prediction(predicted, confidence, severity)

    return {
        "prediction": predicted,
        "label": CLASS_LABELS.get(predicted, predicted.upper()),
        "confidence_estimate": confidence,
        "severity": severity,
        "warning": (
            "Benchmark-model estimate only; this does not prove "
            "an attack occurred."
        ),
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
            error = (
                "Prediction failed. Confirm model.pkl matches "
                "the current feature schema."
            )

    return render_template("index.html", result=result, error=error)


@app.route("/results", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def results():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object containing the named form fields."
        }), 400

    try:
        return jsonify(predict_result(data))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("API prediction failed")
        return jsonify({
            "error": "Prediction failed. Confirm the model matches "
                     "the current feature schema."
        }), 500


@app.route("/history", methods=["GET"])
def history():
    """Return saved prediction history as JSON."""
    if not HISTORY_PATH.exists():
        return jsonify({"history": []})

    try:
        records = pd.read_csv(HISTORY_PATH).replace({np.nan: None})
        return jsonify({"history": records.to_dict(orient="records")})
    except (OSError, pd.errors.ParserError):
        app.logger.exception("Could not read prediction history")
        return jsonify({"error": "Could not read prediction history."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="127.0.0.1", port=port, debug=False)
