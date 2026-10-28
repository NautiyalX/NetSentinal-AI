
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report
)

DATA_DIR = "NSL_Dataset"

COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty"
]

# Original model feature lists, matching train_model.py.
BASE_NUMERIC = [
    "count",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate",
    "dst_host_srv_count",
    "logged_in",
    "same_srv_rate",
    "serror_rate"
]
BASE_CATEGORICAL = ["flag", "service"]

# NOTE:
# The original model also uses "last_flag", but that feature is
# not present in the standard NSL-KDD file. We cannot claim an
# exact 11-feature reproduction without verifying the original
# feature-generation code. This experiment uses 10 available
# baseline features consistently.

LABEL_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos",
    "pod": "dos", "smurf": "dos", "teardrop": "dos",
    "mailbomb": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos",

    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",

    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "phf": "r2l", "multihop": "r2l", "warezmaster": "r2l",
    "warezclient": "r2l", "spy": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "snmpguess": "r2l",
    "snmpgetattack": "r2l", "httptunnel": "r2l",
    "sendmail": "r2l", "named": "r2l",

    "buffer_overflow": "u2r", "loadmodule": "u2r",
    "perl": "u2r", "rootkit": "u2r", "ps": "u2r",
    "sqlattack": "u2r", "xterm": "u2r"
}

def load_dataset(filename):
    df = pd.read_csv(
        f"{DATA_DIR}/{filename}",
        header=None,
        names=COLUMNS
    )

    df["attack"] = df["attack"].astype(str).str.strip().str.lower()
    df["target"] = df["attack"].map(LABEL_MAP)

    if df["target"].isna().any():
        unknown = df.loc[df["target"].isna(), "attack"].unique()
        raise ValueError(f"Unmapped attack labels: {unknown}")

    df = df.drop(columns=["attack", "difficulty"])

    for col in ["protocol_type", "service", "flag"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df

train_df = load_dataset("KDDTrain+.txt")
test_df = load_dataset("KDDTest+.txt")

numeric_features = BASE_NUMERIC
categorical_features = BASE_CATEGORICAL
features = numeric_features + categorical_features

X_train = train_df[features]
y_train = train_df["target"]
X_test = test_df[features]
y_test = test_df["target"]

def make_model(class_weight):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            class_weight=class_weight,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

# Compare the default balanced strategy with stronger weights
# for rare attack classes.
experiments = {
    "A - Balanced subsample": "balanced_subsample",
    "B - Stronger R2L/U2R weights": {
        "normal": 1,
        "dos": 1,
        "probe": 1.5,
        "r2l": 4,
        "u2r": 8
    },
    "C - More aggressive R2L/U2R weights": {
        "normal": 1,
        "dos": 1,
        "probe": 1.5,
        "r2l": 8,
        "u2r": 12
    }
}

labels = ["normal", "dos", "probe", "r2l", "u2r"]

for name, weights in experiments.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    model = make_model(weights)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Macro F1:",
        f"{f1_score(y_test, predictions, labels=labels, average='macro', zero_division=0):.4f}"
    )
    print(
        "Macro Recall:",
        f"{recall_score(y_test, predictions, labels=labels, average='macro', zero_division=0):.4f}"
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    print("\nPer-class recall:")
    for label in labels:
        print(f"{label:8s}: {report[label]['recall']:.4f}")

print("\nExperiments complete. model.pkl was not changed.")
