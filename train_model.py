
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "NSL_Dataset"
TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH = DATA_DIR / "KDDTest+.txt"
MODEL_PATH = BASE_DIR / "model.pkl"

# NSL-KDD contains 41 features, the attack label, and difficulty score.
RAW_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "last_flag",
]

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

DOS = {
    "back", "land", "pod", "neptune", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "worm", "mailbomb",
}
PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
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

    raise ValueError(f"Unknown attack label in dataset: {attack!r}")


def read_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place KDDTrain+.txt and KDDTest+.txt inside NSL_Dataset."
        )

    df = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        sep=",",
        skipinitialspace=True,
    )

    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"{path.name} has an unexpected number of columns: "
            f"{df.shape[1]} (expected {len(RAW_COLUMNS)})."
        )

    df["attack"] = (
        df["attack"].astype(str).str.strip().str.lower().str.rstrip(".")
    )
    df["target"] = df["attack"].map(
        lambda attack: categorize(attack)
    )

    return df


def main():
    print("Loading NSL-KDD datasets...")

    train_df = read_dataset(TRAIN_PATH)
    test_df = read_dataset(TEST_PATH)

    X_train = train_df[FEATURES].copy()
    y_train = train_df["target"]
    X_test = test_df[FEATURES].copy()
    y_test = test_df["target"]

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
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )),
    ])

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")
    print("Training Random Forest classifier...")

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    labels = ["normal", "dos", "probe", "r2l", "u2r"]

    print(f"\nHeld-out accuracy: {accuracy_score(y_test, predicted):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predicted,
        labels=labels,
        zero_division=0,
    ))

    print("Confusion matrix (normal, dos, probe, r2l, u2r):")
    print(confusion_matrix(y_test, predicted, labels=labels))

    # Save to a temporary file first so a failed save won't partially
    # overwrite the existing model.
    temporary_model_path = MODEL_PATH.with_suffix(".tmp")

    try:
        joblib.dump(model, temporary_model_path)
        temporary_model_path.replace(MODEL_PATH)
    finally:
        if temporary_model_path.exists():
            temporary_model_path.unlink()

    print(f"\nSaved retrained model to: {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        raise
