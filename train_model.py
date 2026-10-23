from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "NSL_Dataset"
TRAIN_PATH = DATA_DIR / "Train.txt"
TEST_PATH = DATA_DIR / "Test.txt"
MODEL_PATH = BASE_DIR / "model.pkl"

# Column order follows the existing project's training notebook.
RAW_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "last_flag"
]
NUMERIC_FEATURES = [
    "count", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate", "dst_host_srv_count", "last_flag", "logged_in",
    "same_srv_rate", "serror_rate"
]
CATEGORICAL_FEATURES = ["flag", "service"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DOS = {
    "back", "land", "pod", "neptune", "smurf", "teardrop", "apache2",
    "udpstorm", "processtable", "worm", "mailbomb"
}
PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
    "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
    "httptunnel", "sendmail", "named"
}
U2R = {"buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"}

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
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Keep the original project's NSL_Dataset/Train.txt and Test.txt."
        )
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, sep=",", skipinitialspace=True)
    df["attack"] = df["attack"].astype(str).str.strip().str.lower().str.rstrip(".")
    df["target"] = df["attack"].map(categorize)
    return df

def main():
    train_df = read_dataset(TRAIN_PATH)
    test_df = read_dataset(TEST_PATH)

    X_train = train_df[FEATURES].copy()
    y_train = train_df["target"]
    X_test = test_df[FEATURES].copy()
    y_test = test_df["target"]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
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

    print(f"Training rows: {len(train_df):,}; test rows: {len(test_df):,}")
    print("Inputs exclude attack-name fields to avoid target leakage.")
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    labels = ["normal", "dos", "probe", "r2l", "u2r"]
    print(f"\nHeld-out accuracy: {accuracy_score(y_test, predicted):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predicted, labels=labels, zero_division=0))
    print("Confusion matrix (labels: normal, dos, probe, r2l, u2r):")
    print(confusion_matrix(y_test, predicted, labels=labels))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved retrained model to {MODEL_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        raise
