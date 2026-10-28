
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "NSL_Dataset" / "KDDTrain+.txt"

COLUMNS = [
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

LABELS = ["normal", "dos", "probe", "r2l", "u2r"]

BASE_NUMERIC = [
    "count", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_same_srv_rate",
    "dst_host_srv_count", "last_flag", "logged_in",
    "same_srv_rate", "serror_rate",
]
BASE_CATEGORICAL = ["flag", "service"]

ALL_NUMERIC = [
    c for c in COLUMNS
    if c not in {"protocol_type", "service", "flag", "attack"}
]
ALL_CATEGORICAL = ["protocol_type", "service", "flag"]


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
    raise ValueError(f"Unknown attack label: {attack}")


def make_model(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )),
    ])


def evaluate(name, model, X_train, X_val, y_train, y_val):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    print(f"Validation accuracy: {accuracy_score(y_val, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_val, predictions, labels=LABELS, average='macro', zero_division=0):.4f}")
    print(f"Macro recall: {recall_score(y_val, predictions, labels=LABELS, average='macro', zero_division=0):.4f}")
    print("\nPer-class report:")
    print(classification_report(
        y_val, predictions, labels=LABELS, zero_division=0
    ))


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH, header=None, names=COLUMNS,
        sep=",", skipinitialspace=True,
    )
    df["attack"] = (
        df["attack"].astype(str).str.strip().str.lower().str.rstrip(".")
    )
    df["target"] = df["attack"].map(categorize)

    # The same split is used for both candidates.
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    y_train = train_df["target"]
    y_val = val_df["target"]

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print("\nValidation class counts:")
    print(y_val.value_counts().reindex(LABELS, fill_value=0))

    baseline_features = BASE_NUMERIC + BASE_CATEGORICAL
    expanded_features = ALL_NUMERIC + ALL_CATEGORICAL

    baseline = make_model(BASE_NUMERIC, BASE_CATEGORICAL)
    expanded = make_model(ALL_NUMERIC, ALL_CATEGORICAL)

    evaluate(
        "BASELINE: 11 features",
        baseline,
        train_df[baseline_features],
        val_df[baseline_features],
        y_train,
        y_val,
    )

    evaluate(
        "EXPANDED: 40 features",
        expanded,
        train_df[expanded_features],
        val_df[expanded_features],
        y_train,
        y_val,
    )

    print("\nComparison finished. Existing model.pkl was not changed.")
    print("Choose primarily by macro F1 and rare-class recall,")
    print("not accuracy alone. Validation results are not final test results.")


if __name__ == "__main__":
    main()
