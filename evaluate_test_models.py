
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)
from imblearn.ensemble import BalancedRandomForestClassifier

TRAIN_PATH = "NSL_Dataset/KDDTrain+.txt"
TEST_PATH = "NSL_Dataset/KDDTest+.txt"

COLUMNS = [
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
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

DOS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "mailbomb", "apache2", "processtable", "udpstorm",
}
PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop",
    "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
    "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named",
}
CLASS_ORDER = ["normal", "dos", "probe", "r2l", "u2r"]

FEATURES = [
    "count",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_same_srv_rate",
    "dst_host_srv_count",
    "logged_in",
    "same_srv_rate",
    "serror_rate",
    "flag",
    "service",
]


def map_label(label):
    if label == "normal":
        return "normal"
    if label in DOS:
        return "dos"
    if label in PROBE:
        return "probe"
    if label in R2L:
        return "r2l"
    return "u2r"


def load_data(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["target"] = df["label"].apply(map_label)
    X = pd.get_dummies(
        df[FEATURES], columns=["flag", "service"]
    ).astype(float)
    return X, df["target"]


print("Loading datasets...")
X_train, y_train = load_data(TRAIN_PATH)
X_test, y_test = load_data(TEST_PATH)

# Align test columns to training columns.
# Unknown test-only categories become all-zero training columns.
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Reserve a validation split from KDDTrain+ to select the threshold.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

print("Training Balanced Random Forest...")
model = BalancedRandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_fit, y_fit)

# Select the U2R threshold using validation data only.
val_probabilities = model.predict_proba(X_val)
classes = list(model.classes_)
u2r_index = classes.index("u2r")

val_baseline = model.predict(X_val)
val_threshold_predictions = val_baseline.copy()
val_threshold_predictions[
    val_probabilities[:, u2r_index] >= 0.40
] = "u2r"

print("\nValidation macro F1:")
print(
    "Baseline:",
    round(f1_score(y_val, val_baseline, average="macro", zero_division=0), 4),
)
print(
    "U2R threshold 0.40:",
    round(
        f1_score(
            y_val,
            val_threshold_predictions,
            average="macro",
            zero_division=0,
        ),
        4,
    ),
)

# Only now evaluate on KDDTest+.
test_probabilities = model.predict_proba(X_test)
test_baseline = model.predict(X_test)
test_threshold = test_baseline.copy()
test_threshold[test_probabilities[:, u2r_index] >= 0.40] = "u2r"


def evaluate(name, predictions):
    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Macro F1:",
        round(f1_score(y_test, predictions, average="macro", zero_division=0), 4),
    )
    print(
        "Macro Recall:",
        round(recall_score(y_test, predictions, average="macro", zero_division=0), 4),
    )
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASS_ORDER,
            zero_division=0,
        )
    )
    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=CLASS_ORDER))

    actual_u2r = y_test == "u2r"
    predicted_u2r = predictions == "u2r"

    print("\nU2R details:")
    print(
        "Precision:",
        round(precision_score(actual_u2r, predicted_u2r, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(actual_u2r, predicted_u2r, zero_division=0), 4),
    )
    print("False positives:", int((~actual_u2r & predicted_u2r).sum()))
    print("False negatives:", int((actual_u2r & ~predicted_u2r).sum()))


evaluate("KDDTEST+ BASELINE", test_baseline)
evaluate("KDDTEST+ U2R THRESHOLD 0.40", test_threshold)

print("\nEvaluation complete. model.pkl was not changed.")
