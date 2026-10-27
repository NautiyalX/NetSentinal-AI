# 🔐 NetSentinel — Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange?style=flat-square)
![Cybersecurity](https://img.shields.io/badge/Domain-Network%20Security-red?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-lightblue?style=flat-square)
![Flask](https://img.shields.io/badge/Framework-Flask-green?style=flat-square&logo=flask)

**NetSentinel** is a machine learning-based Network Intrusion Detection System that analyzes network connection statistics and classifies traffic into normal traffic or one of four broad attack categories. It provides a web interface for predictions, confidence estimates, severity levels, and prediction history.

The project uses a Random Forest classifier trained on the NSL-KDD benchmark dataset and a Flask backend for prediction and validation.

---

## 📌 Problem Statement

Network attacks can disrupt services, compromise sensitive information, and affect system availability. Traditional security systems often rely on predefined signatures and rules to identify known threats.

NetSentinel explores how machine learning can help classify network traffic using connection-level and traffic-statistical features.

The system aims to:

- Classify network activity into normal or malicious traffic categories.
- Identify broad attack categories using a trained machine learning model.
- Present prediction results through a simple web interface.
- Display model confidence estimates and severity levels.
- Maintain a history of previous predictions.

**Scope:** NetSentinel is currently a benchmark-based demonstration. Users enter network connection statistics manually; the application does not capture live network packets.

---

## 🎯 Key Features

### 1. Machine Learning-Based Detection
- Random Forest classification model.
- Trained using the NSL-KDD training dataset.
- Evaluated against a separate NSL-KDD test dataset.
- Five output classes: Normal, DoS, Probe, R2L, and U2R.

### 2. Prediction Confidence
Displays the model's maximum predicted class probability as a percentage alongside the prediction.

For example, a prediction may display `Probe / scanning` with a confidence estimate of `35.61%`.

This value is a model estimate, not a guarantee that the prediction is correct.

### 3. Attack Severity
Each predicted class is assigned a severity level.

| Predicted class | Description | Severity |
|---|---|---|
| Normal | Legitimate traffic category | Low |
| DoS | Denial of Service | High |
| Probe | Scanning or reconnaissance | Medium |
| R2L | Remote-to-Local attack | High |
| U2R | User-to-Root privilege escalation | Critical |

### 4. Prediction History
Successful predictions are recorded in `prediction_history.csv`, including:

- Timestamp
- Predicted class
- Confidence percentage
- Severity level

The `/history` endpoint returns stored prediction records in JSON format.

### 5. Input Validation
The Flask backend validates numeric inputs, rejects non-finite values, checks rate features are between 0 and 1, and validates integer and login-status fields.

### 6. Web Interface and API
- Dark-themed Flask web interface.
- Form-based network feature input.
- Prediction results with severity and confidence.
- JSON prediction endpoints for API testing.

---

## 🧠 Attack Categories

NetSentinel groups traffic into five classes.

| Class | Description | Examples of associated attacks |
|---|---|---|
| Normal | Legitimate network activity | Normal connections |
| DoS | Attempts to overwhelm network or system resources | Neptune, Smurf, Teardrop |
| Probe | Scanning and reconnaissance activity | Nmap, Portsweep, Ipsweep |
| R2L | Remote-to-Local unauthorized access | Password guessing, FTP Write |
| U2R | User-to-Root privilege escalation | Buffer Overflow, Rootkit |

These are broad categories based on the dataset's attack labels. A model prediction is not proof that an actual attack has occurred.

---

## 📊 Dataset

**Dataset:** NSL-KDD

NSL-KDD is a benchmark dataset used in network intrusion detection research. It contains labeled connection records and 41 original traffic features, followed by an attack label and a difficulty value.

### Dataset files

The project uses these files:

- `NSL_Dataset/KDDTrain+.txt` — training records.
- `NSL_Dataset/KDDTest+.txt` — held-out test records.

The files are included in the repository's dataset folder.

### Feature preprocessing

The training pipeline uses selected features rather than all 41 original features.

**Numeric features:**
- `count`
- `dst_host_diff_srv_rate`
- `dst_host_same_src_port_rate`
- `dst_host_same_srv_rate`
- `dst_host_srv_count`
- `last_flag`
- `logged_in`
- `same_srv_rate`
- `serror_rate`

**Categorical features:**
- `flag`
- `service`

Numeric missing values are handled using median imputation. Categorical values are handled using most-frequent imputation and one-hot encoding.

Attack labels are mapped into the five output classes.

---

## 🏗️ Project Architecture

```text
NetSentinal-AI/
├── NSL_Dataset/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── templates/
│   └── index.html
├── app.py
├── train_model.py
├── model.pkl
├── prediction_history.csv    # Generated locally at runtime
├── test_api.py               # API validation test script
├── Network Intrusion Detection System.ipynb
├── corrm.csv
├── num_summary.csv
├── requirements.txt
├── Dockerfile
├── Procfile
├── .gitignore
├── README.md
└── LICENSE
```

`prediction_history.csv` is generated when predictions are made and is excluded from version control.

### Application workflow

```text
User enters connection statistics
             |
             v
      Flask web interface
             |
             v
       Input validation
             |
             v
   Preprocessing pipeline
             |
             v
     Random Forest model
             |
             v
  Predicted class and probability
             |
             v
   Severity classification
             |
             v
    Display prediction result
             |
             v
   Save prediction to CSV history
```

---

## 🤖 Model Training and Evaluation

The model is trained by `train_model.py`.

### Training configuration

- Algorithm: Random Forest Classifier
- Number of trees: 250
- Random state: 42
- Class weighting: `balanced_subsample`
- Minimum samples per leaf: 2
- Preprocessing: Scikit-learn pipelines with numeric and categorical transformations

The complete preprocessing and classification pipeline is serialized into `model.pkl` using Joblib.

### Current evaluation results

The latest training run produced the following results on the held-out test dataset.

| Metric | Result |
|---|---:|
| Training records | 125,973 |
| Test records | 22,544 |
| Overall accuracy | 78.40% |
| Weighted F1-score | 0.77 |
| Macro F1-score | 0.63 |

### Per-class performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Normal | 0.69 | 0.97 | 0.81 |
| DoS | 0.96 | 0.80 | 0.88 |
| Probe | 0.80 | 0.63 | 0.70 |
| R2L | 0.97 | 0.25 | 0.40 |
| U2R | 0.37 | 0.37 | 0.37 |

The results show that performance varies by attack category. In particular, R2L and U2R recall require improvement. These results are from the current benchmark evaluation and should not be interpreted as real-world detection performance.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model serialization | Joblib |
| Backend | Flask |
| Frontend | HTML, CSS |
| Data storage | CSV prediction history |
| Dataset | NSL-KDD |
| Development environment | VS Code |
| Version control | Git and GitHub |
| Container configuration | Dockerfile |

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/NautiyalX/NetSentinal-AI.git
cd NetSentinal-AI
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Verify the project files

Ensure the following files exist:

```text
model.pkl
app.py
templates/index.html
NSL_Dataset/KDDTrain+.txt
NSL_Dataset/KDDTest+.txt
```

The existing trained model is sufficient to run predictions. Retraining is optional.

### 5. Start the Flask application

```powershell
python app.py
```

Open the application at:

**http://127.0.0.1:10000/**

### 6. Retrain the model (optional)

To retrain the model using the included dataset:

```powershell
python train_model.py
```

This evaluates the trained pipeline and saves the resulting model to `model.pkl`.

---

## 🔌 API Endpoints

### 1. Web application

`GET /`

Opens the main prediction interface.

### 2. Prediction API

`POST /api/predict`

Accepts a JSON object containing the required feature values.

`POST /results`

Provides the same JSON prediction functionality.

Example request:

```json
{
  "service": "http",
  "flag": "SF",
  "count": 10,
  "dst_host_srv_count": 10,
  "dst_host_diff_srv_rate": 0.2,
  "dst_host_same_src_port_rate": 0.5,
  "dst_host_same_srv_rate": 0.8,
  "same_srv_rate": 0.8,
  "serror_rate": 0.1,
  "last_flag": 10,
  "logged_in": 0
}
```

Example response:

```json
{
  "prediction": "probe",
  "label": "Probe / scanning",
  "confidence_estimate": 0.3561,
  "severity": "Medium",
  "warning": "Benchmark-model estimate only; this does not prove an attack occurred."
}
```

The prediction and confidence shown above are illustrative.

### 3. Prediction history

`GET /history`

Returns previously saved prediction records as JSON.

---

## 🔒 Security and Limitations

- NetSentinel uses benchmark data rather than live network packet capture.
- Predictions depend on manually supplied feature values.
- The model has not been established as production-ready for enterprise or defense networks.
- The confidence estimate is not necessarily a calibrated probability of correctness.
- The current model has lower recall for R2L and U2R classes.
- Prediction history is stored locally in CSV format and is not a centralized database.
- The application should be deployed with appropriate security controls before being exposed to an external network.

---

## 💡 Future Improvements

- [ ] Improve recall for R2L and U2R attack categories.
- [ ] Add explainable AI using SHAP to explain individual predictions.
- [ ] Add visual analytics for prediction history and attack distributions.
- [ ] Introduce automated tests for the model and API.
- [ ] Evaluate performance on newer datasets such as CICIDS2017 or UNSW-NB15.
- [ ] Explore real-time packet capture using Scapy or PyShark.
- [ ] Add experiment tracking using MLflow.
- [ ] Explore sequential deep-learning approaches for network traffic analysis.

---

## 📚 Project Summary

NetSentinel demonstrates a complete machine learning workflow for network intrusion classification, from dataset preprocessing and model training to Flask-based inference and prediction history.

The current implementation achieves **78.40% accuracy on the NSL-KDD held-out test set** and includes five-class prediction, confidence estimates, severity classification, input validation, and CSV-based history.

The project is intended for educational purposes and further experimentation in machine learning-based network security.

---

**Repository:** [NautiyalX/NetSentinal-AI](https://github.com/NautiyalX/NetSentinal-AI)
