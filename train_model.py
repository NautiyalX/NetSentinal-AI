
from pathlib import Path
import shutil
import sys

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.ensemble import BalancedRandomForestClassifier


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "NSL_Dataset"

TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH = DATA_DIR / "KDDTest+.txt"

MODEL_PATH = BASE_DIR / "model.pkl"
BACKUP_PATH = BASE_DIR / "model_backup.pkl"
TEMP_PATH = BASE_DIR / "model.tmp"


# --------------------------------------------------
# 2. NSL-KDD dataset columns
# 41 features + attack label + difficulty metadata
# --------------------------------------------------
RAW_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "difficulty",
]


# --------------------------------------------------
# 3. Features — must match app.py exactly
# --------------------------------------------------
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

CLASS_ORDER = ["normal", "dos", "probe", "r2l", "u2r"]


# --------------------------------------------------
# 4. Map individual attacks into five categories
# --------------------------------------------------
DOS = {
    "back", "land", "pod", "neptune", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "worm", "mailbomb",
}

PROBE = {
    "satan", "ipsweep", "nmap", "portsweep", "mscan", "saint",
}

R2L = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop",
    "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
    "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named",
}

U2R = {
    "buffer_overflow", "loadmodule", "rootkit", "perl",
    "sqlattack", "xterm", "ps",
}


def categorize(attack):
    attack = str(attack).strip().lower().rstrip(".")

    if attack == "normal":
        return "normal"
    if attack in DOS:
        return "dos"
    if attack in PROBE:
        return "probe"
    if attack in R2L:
        return "r2l"
    if attack in U2R:
        return "u2r"

    raise ValueError(f"Unknown attack label: {attack!r}")


# --------------------------------------------------
# 5. Load dataset
# --------------------------------------------------
def read_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        sep=",",
        skipinitialspace=True,
    )

    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(RAW_COLUMNS)} columns, "
            f"found {df.shape[1]}."
        )

    df["attack"] = (
        df["attack"].astype(str).str.strip().str.lower().str.rstrip(".")
    )
    df["target"] = df["attack"].apply(categorize)

    # Difficulty is metadata, not a model input.
    X = df[FEATURES].copy()
    y = df["target"].copy()

    return X, y


# --------------------------------------------------
# 6. Train, evaluate, and safely save
# --------------------------------------------------
def main():
    print("Loading NSL-KDD datasets...")

    X_train, y_train = read_dataset(TRAIN_PATH)
    X_test, y_test = read_dataset(TEST_PATH)

    print("Training rows:", f"{len(X_train):,}")
    print("Testing rows: ", f"{len(X_test):,}")
    print("Input features:", len(FEATURES))
    print("Feature names:", FEATURES)

    print("\nTraining class distribution:")
    print(y_train.value_counts().to_string())

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", BalancedRandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("\nTraining Balanced Random Forest...")
    model.fit(X_train, y_train)

    # Evaluate on the official test set.
    print("\nEvaluating on KDDTest+...")
    predictions = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Macro F1:",
        round(f1_score(
            y_test, predictions, average="macro", zero_division=0
        ), 4),
    )
    print(
        "Macro Recall:",
        round(recall_score(
            y_test, predictions, average="macro", zero_division=0
        ), 4),
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        zero_division=0,
    ))

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_ORDER,
    ))

    # Back up the existing model before replacing it.
    if MODEL_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"\nExisting model backed up to: {BACKUP_PATH}")

    # Save to a temporary file, then replace the model atomically.
    try:
        joblib.dump(model, TEMP_PATH)
        joblib.load(TEMP_PATH)  # Verify that the saved file loads.
        TEMP_PATH.replace(MODEL_PATH)
    finally:
        if TEMP_PATH.exists():
            TEMP_PATH.unlink()

    print(f"\nNew model saved to: {MODEL_PATH}")
    print("Model input features:", len(FEATURES))
    print("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        raise
