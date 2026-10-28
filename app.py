
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Resolve all paths relative to this app.py file.
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "model.pkl"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# These must match train_model.py.
NUMERIC_FEATURES = [
    "count",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate",
    "dst_host_srv_count",
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

ALLOWED_FLAGS = {
    "sf", "s0", "rej", "rstr", "s1", "s2", "s3",
    "sh", "rsto", "oth", "rstos0", "rsts0",
}


# Prevent the browser from reusing an outdated HTML response.
@app.after_request
def disable_caching(response):
    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def load_model():
    if not MODEL_PATH.is_file():
        app.logger.warning("Model file not found: %s", MODEL_PATH)
        return None

    loaded = joblib.load(MODEL_PATH)

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

    for name in NUMERIC_FEATURES:
        value = data.get(name)

        if value is None or str(value).strip() == "":
            raise ValueError(f"Please enter {name.replace('_', ' ')}.")

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name.replace('_', ' ')} must be numeric.")

        if not np.isfinite(number):
            raise ValueError(f"{name.replace('_', ' ')} must be finite.")

        cleaned[name] = number

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

    for name in ("count", "dst_host_srv_count"):
        if cleaned[name] < 0 or not cleaned[name].is_integer():
            raise ValueError(
                f"{name.replace('_', ' ')} must be a "
                "non-negative whole number."
            )
        cleaned[name] = int(cleaned[name])

    if cleaned["logged_in"] not in (0, 1):
        raise ValueError("Login status must be 0 (No) or 1 (Yes).")

    cleaned["logged_in"] = int(cleaned["logged_in"])

    for name in CATEGORICAL_FEATURES:
        value = str(data.get(name, "")).strip().lower()

        if not value:
            raise ValueError(f"Please select {name}.")

        if name == "flag" and value not in ALLOWED_FLAGS:
            raise ValueError("Please select a valid NSL-KDD connection flag.")

        cleaned[name] = value

    return pd.DataFrame(
        [{name: cleaned[name] for name in FEATURES}],
        columns=FEATURES,
    )


def save_prediction(prediction, confidence, severity):
    record = pd.DataFrame([{
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
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
            "The model is unavailable or its feature schema is incompatible. "
            "Check model.pkl and the training feature schema."
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
            error = "Prediction failed. Check the model and input schema."

    response = app.make_response(
        render_template("index.html", result=result, error=error)
    )

    # Temporary diagnostic headers to verify the served template.
    response.headers["X-NetSentinel-App"] = str(Path(__file__).resolve())
    response.headers["X-NetSentinel-Template"] = str(TEMPLATE_DIR / "index.html")
    return response


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
            "error": "Prediction failed. Check the model and input schema."
        }), 500


@app.route("/history", methods=["GET"])
def history():
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

    print(f"NetSentinel app: {Path(__file__).resolve()}")
    print(f"Template file: {TEMPLATE_DIR / 'index.html'}")
    print(f"Model file: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")

    app.run(host="127.0.0.1", port=port, debug=False)
