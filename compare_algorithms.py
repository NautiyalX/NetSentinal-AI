
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report
)

from imblearn.ensemble import BalancedRandomForestClassifier


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

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

# Same 10 available features used in our recent experiments.
NUMERIC = [
    "count",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate",
    "dst_host_srv_count",
    "logged_in",
    "same_srv_rate",
    "serror_rate"
]

CATEGORICAL = ["flag", "service"]
FEATURES = NUMERIC + CATEGORICAL

LABELS = ["normal", "dos", "probe", "r2l", "u2r"]


# --------------------------------------------------
# 2. Attack label mapping
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

def load_dataset(filename):
    df = pd.read_csv(
        f"{DATA_DIR}/{filename}",
        header=None,
        names=COLUMNS
    )

    df["attack"] = (
        df["attack"].astype(str).str.strip().str.lower()
    )

    df["target"] = df["attack"].map(LABEL_MAP)

    if df["target"].isna().any():
        unknown = df.loc[df["target"].isna(), "attack"].unique()
        raise ValueError(f"Unmapped attack labels: {unknown}")

    df = df.drop(columns=["attack", "difficulty"])

    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df


train_df = load_dataset("KDDTrain+.txt")
test_df = load_dataset("KDDTest+.txt")

X_train = train_df[FEATURES]
y_train = train_df["target"]

X_test = test_df[FEATURES]
y_test = test_df["target"]

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Input features:", len(FEATURES))
print("Feature names:", FEATURES)
print("Class order:", LABELS)


# --------------------------------------------------
# 4. Preprocessing pipeline
# --------------------------------------------------

def build_pipeline(classifier):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", classifier)
    ])


# --------------------------------------------------
# 5. Models to compare
# --------------------------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=250,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Balanced Random Forest": BalancedRandomForestClassifier(
        n_estimators=250,
        sampling_strategy="all",
        replacement=True,
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------

results = []

for name, classifier in models.items():
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    model = build_pipeline(classifier)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )
    macro_recall = recall_score(
        y_test, predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")

    print("\nPer-class classification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=LABELS,
        zero_division=0
    ))

    # Rows = actual class, columns = predicted class.
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS
    )

    print("\nConfusion matrix")
    print("Rows = actual; columns = predicted")
    print("Class order:", LABELS)
    print(pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in LABELS],
        columns=[f"Predicted {label}" for label in LABELS]
    ).to_string())

    # For each class, calculate one-vs-rest errors.
    print("\nPer-class false positives and false negatives:")

    total = cm.sum()

    for i, label in enumerate(LABELS):
        true_positive = cm[i, i]
        false_negative = cm[i, :].sum() - true_positive
        false_positive = cm[:, i].sum() - true_positive
        true_negative = total - (
            true_positive + false_negative + false_positive
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0
        )

        print(
            f"{label:8s} | "
            f"TP={true_positive:5d} "
            f"FP={false_positive:5d} "
            f"FN={false_negative:5d} "
            f"TN={true_negative:5d} | "
            f"Precision={precision:.4f} "
            f"Recall={recall:.4f}"
        )

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Macro Recall": macro_recall,
        "R2L Recall": report["r2l"]["recall"],
        "U2R Recall": report["u2r"]["recall"]
    })


# --------------------------------------------------
# 7. Final comparison
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print(
    pd.DataFrame(results)
    .sort_values("Macro F1", ascending=False)
    .to_string(index=False)
)

print("\nComparison finished. Existing model.pkl was not modified.")
