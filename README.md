# NetSentinel AI — Network Intrusion Detection System

**An ML-powered network traffic classification system for identifying potentially suspicious network activity.**

NetSentinel AI is a machine-learning project that analyzes network connection statistics and classifies traffic into five broad categories: Normal, Denial of Service (DoS), Probe, Remote-to-Local (R2L), and User-to-Root (U2R).

The project combines a Flask web application, a trained machine-learning pipeline, and a dark-themed dashboard to demonstrate network intrusion detection using the NSL-KDD dataset.

## Key Features

- **Network Traffic Classification:** Predicts one of five traffic categories from connection statistics.
- **Machine Learning Pipeline:** Uses a Balanced Random Forest classifier with preprocessing for numerical and categorical features.
- **Interactive Web Interface:** Submit connection statistics through a browser-based dashboard.
- **Prediction Confidence Estimate:** Displays the model's maximum predicted class probability.
- **Severity Indicators:** Assigns an indicative severity level to each predicted category.
- **Prediction History:** Records predictions and timestamps in a local CSV file.
- **JSON Prediction API:** Supports programmatic predictions through Flask endpoints.
- **Model Evaluation Tools:** Includes scripts for comparing algorithms, evaluating test performance, and experimenting with thresholds and class weights.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web framework | Flask |
| Machine learning | scikit-learn, imbalanced-learn |
| Data processing | Pandas, NumPy |
| Model serialization | Joblib |
| Frontend | HTML, CSS, JavaScript |
| Dataset | NSL-KDD |
| Prediction history | CSV |

## System Architecture

```text
User
  |
  v
Flask Web Interface
  |
  v
Input Validation
  |
  v
Saved ML Pipeline (model.pkl)
  |
  v
Balanced Random Forest
  |
  v
Predicted Traffic Category
  |
  +----> Result and confidence estimate
  |
  +----> Prediction history (CSV)
```

The Flask application receives connection statistics, validates the inputs, passes them to the saved preprocessing and classification pipeline, and displays the prediction.

## Dataset

NetSentinel uses the NSL-KDD intrusion detection dataset, a benchmark dataset derived from KDD Cup 1999.

The training script expects these files:

```text
NSL_Dataset/
├── KDDTrain+.txt
└── KDDTest+.txt
```

The dataset contains network connection records, attack labels, and a difficulty score. The difficulty score is excluded from the model inputs.

The project maps attack labels into five categories:

| Class | Description |
|---|---|
| `normal` | Normal network traffic |
| `dos` | Denial-of-Service attacks |
| `probe` | Network probing and scanning |
| `r2l` | Remote-to-Local attacks |
| `u2r` | User-to-Root attacks |

**Dataset note:** Download the NSL-KDD dataset separately if it is not included in your local project copy. Place the required files in `NSL_Dataset/` before training.

## Model Features

The current model uses **10 input features**.

### Numerical features

| Feature | Description |
|---|---|
| `count` | Number of connections to the same destination host within the dataset's time window |
| `dst_host_diff_srv_rate` | Rate of connections to different services for the destination host |
| `dst_host_same_src_port_rate` | Rate of connections using the same source port for the destination host |
| `dst_host_same_srv_rate` | Rate of connections using the same service for the destination host |
| `dst_host_srv_count` | Number of connections to the same service for the destination host |
| `logged_in` | Whether the connection successfully logged in |
| `same_srv_rate` | Rate of connections using the same service |
| `serror_rate` | Rate of connections with SYN-related errors |

### Categorical features

| Feature | Description |
|---|---|
| `flag` | Connection status or state, such as `SF`, `S0`, or `REJ` |
| `service` | Network service associated with the connection, such as HTTP |

The preprocessing pipeline handles missing numerical values and encodes categorical values before classification.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/NautiyalX/NetSentinal-AI.git
cd NetSentinal-AI
```

To use the latest work on the development branch:

```bash
git checkout update-netsentinel
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python executable directly as shown in the next steps.

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the project files

Ensure that `model.pkl` exists in the project root and that the dataset files are present if you intend to retrain the model.

## Running NetSentinel

Start the Flask application from the project root:

```bash
python app.py
```

On Windows, if using the virtual environment directly:

```powershell
.\venv\Scripts\python.exe app.py
```

Open the application in your browser:

**http://127.0.0.1:10000/**

Enter the network connection statistics and select **Analyze connection** to view the prediction.

## Training the Model

To train the model using the available NSL-KDD training and test files, run:

```bash
python train_model.py
```

The training pipeline:

1. Loads and preprocesses the NSL-KDD dataset.
2. Maps attack labels into five broad classes.
3. Uses the defined 10 input features.
4. Trains a Balanced Random Forest classifier.
5. Evaluates predictions on the official KDDTest+ dataset.
6. Saves the trained pipeline as `model.pkl`.

**Warning:** Training replaces the active model after a successful run. Keep a separate backup if you want to preserve a previous version.

## Model Evaluation

The current training run reported the following results on the official KDDTest+ dataset:

| Metric | Result |
|---|---:|
| Accuracy | 74.41% |
| Macro F1-score | 0.5599 |
| Macro Recall | 0.6276 |

The model has uneven performance across classes. In particular, R2L and U2R attacks are more difficult to classify reliably, and U2R predictions have a high false-positive risk.

Accuracy alone does not describe intrusion detection performance adequately. Macro F1, per-class precision and recall, and the confusion matrix should also be considered.

### Evaluation and experimentation scripts

The repository includes additional scripts for model analysis:

| Script | Purpose |
|---|---|
| `compare_algorithms.py` | Compare classification algorithms |
| `compare_models.py` | Additional model comparison experiments |
| `evaluate_test_models.py` | Evaluate candidate models on test data |
| `tune_class_weights.py` | Experiment with class weighting |
| `tune_thresholds.py` | Experiment with prediction thresholds |
| `test_api.py` | Test the prediction API |

Run these scripts from the project root using your Python environment. They are experimental tools; review their configuration before running them, especially when using the official test dataset.

## API Usage

NetSentinel provides a JSON prediction endpoint at:

```text
POST /api/predict
```

Send a JSON object containing the named input features.

Example request:

```json
{
  "count": 10,
  "dst_host_diff_srv_rate": 0.05,
  "dst_host_same_src_port_rate": 0.02,
  "dst_host_same_srv_rate": 0.95,
  "dst_host_srv_count": 50,
  "logged_in": 1,
  "same_srv_rate": 0.95,
  "serror_rate": 0.0,
  "flag": "sf",
  "service": "http"
}
```

Example using `curl`:

```bash
curl -X POST http://127.0.0.1:10000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"count\":10,\"dst_host_diff_srv_rate\":0.05,\"dst_host_same_src_port_rate\":0.02,\"dst_host_same_srv_rate\":0.95,\"dst_host_srv_count\":50,\"logged_in\":1,\"same_srv_rate\":0.95,\"serror_rate\":0.0,\"flag\":\"sf\",\"service\":\"http\"}"
```

The response includes the predicted class, a human-readable label, a confidence estimate when supported by the model, and an indicative severity level.

Additional endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Display the dashboard |
| `/` | POST | Submit a prediction from the web form |
| `/api/predict` | POST | Return a prediction as JSON |
| `/results` | POST | Alternative JSON prediction endpoint |
| `/history` | GET | Retrieve recorded prediction history |

## Prediction History

Predictions are recorded in:

```text
prediction_history.csv
```

The file contains the prediction timestamp, predicted class, confidence estimate, and severity. It is generated locally when predictions are made.

Avoid committing generated history files if they contain data you do not want to publish.

## Project Structure

```text
NetSentinal-AI/
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── NSL_Dataset/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── templates/
│   └── index.html
├── static/
├── results/
├── compare_algorithms.py
├── compare_models.py
├── evaluate_test_models.py
├── test_api.py
├── tune_class_weights.py
├── tune_thresholds.py
├── Dockerfile
└── README.md
```

The exact files and folders may vary depending on the version of the repository. The dataset, generated model, and prediction history may be maintained separately from version control.

## Limitations

- The model uses a selected subset of 10 NSL-KDD features rather than all original dataset features.
- The input values supplied manually through the web interface may not reflect actual live network traffic.
- NSL-KDD is a benchmark dataset and may not represent modern production network environments.
- The confidence estimate is the model's predicted class probability, not a guarantee that a prediction is correct.
- The application currently demonstrates classification; it does not automatically capture live packets or block malicious connections.
- Model performance must be independently validated before any real-world security deployment.

**NetSentinel should be treated as an educational and experimental intrusion detection project, not as a production-ready security monitoring solution.**

## Future Improvements

- Integrate live network traffic capture and feature extraction.
- Improve detection of rare attack classes.
- Add confusion-matrix visualizations and per-class performance dashboards.
- Introduce alerting and filtering for suspicious predictions.
- Add authentication and persistent storage for multi-user deployments.
- Evaluate the model on newer datasets and representative real-world traffic.
- Containerize the application for easier deployment.

## Contributing

Contributions and suggestions are welcome. For significant changes, create a separate branch, test the changes locally, and submit a pull request.

## Disclaimer

This project is intended for educational, research, and defensive cybersecurity purposes. Predictions are estimates and should not be used as the sole basis for security decisions.

## Author

**Adarsh Nautiyal**

GitHub: [@NautiyalX](https://github.com/NautiyalX)

Project: [NetSentinal-AI](https://github.com/NautiyalX/NetSentinal-AI)
