
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.ensemble import BalancedRandomForestClassifier

DATA_PATH = "NSL_Dataset/KDDTrain+.txt"

columns = [
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

df = pd.read_csv(DATA_PATH, header=None, names=columns)

dos = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "mailbomb", "apache2", "processtable", "udpstorm",
}
probe = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
r2l = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop",
    "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
    "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named",
}

def map_label(label):
    if label == "normal":
        return "normal"
    if label in dos:
        return "dos"
    if label in probe:
        return "probe"
    if label in r2l:
        return "r2l"
    return "u2r"

df["target"] = df["label"].apply(map_label)

features = [
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

X = pd.get_dummies(df[features], columns=["flag", "service"]).astype(float)
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = BalancedRandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_val)
classes = list(model.classes_)
baseline = model.predict(X_val)
results = []

def record(name, predictions, target_class=None):
    row = {
        "Experiment": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Macro_F1": f1_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "Macro_Recall": recall_score(
            y_val, predictions, average="macro", zero_division=0
        ),
    }

    if target_class:
        actual = (y_val == target_class)
        predicted = (predictions == target_class)
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())

        row.update({
            "Class": target_class,
            "Precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "Class_F1": f1_score(
                actual, predicted, zero_division=0
            ),
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
        })

    results.append(row)

record("Baseline", baseline)

for target_class, thresholds in [
    ("r2l", [0.30, 0.40, 0.50, 0.60, 0.70]),
    ("u2r", [0.30, 0.40, 0.50, 0.60, 0.70]),
]:
    index = classes.index(target_class)

    for threshold in thresholds:
        predictions = baseline.copy()
        predictions[probabilities[:, index] >= threshold] = target_class

        record(
            f"{target_class.upper()} threshold {threshold:.2f}",
            predictions,
            target_class,
        )

result_df = pd.DataFrame(results)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

print("\nTHRESHOLD EXPERIMENT RESULTS")
print(result_df.to_string(index=False))

print("\nExperiment complete. model.pkl was not changed.")
