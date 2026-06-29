# Hate Speech Detection - MLOps Project Report

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for detecting hate speech in tweets. It covers data preprocessing, model training, experiment tracking, model registration, REST API development, UI development, and containerization using Docker.

---

## 2. Project Structure

```
hate-speech-mlops/
├── data/
│   └── raw/
│       └── labeled_data.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── svm_model.pkl
│   └── rf_model.pkl
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── mlruns/
├── app.py
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── mlflow.db
```

---

## 3. Dataset

- **Source:** `labeled_data.csv`
- **Original Labels:**
  - 0 → Hate Speech
  - 1 → Offensive Language
  - 2 → Neither
- **Binary Mapping:**
  - 1 → Hate Speech
  - 0 → Not Hate Speech

---

## 4. Configuration (`src/config.py`)

Central configuration file for all paths and hyperparameters.

| Parameter | Value |
|-----------|-------|
| TEST_SIZE | 0.3 |
| CV_SIZE | 0.3 |
| RANDOM_STATE | 35 |
| SVM Kernel | linear |
| SVM C | 1.0 |
| RF n_estimators | 99 |
| RF random_state | 42 |
| MLflow Experiment | hate-speech-detection |

---

## 5. Preprocessing Pipeline (`src/preprocessing.py`)

Steps applied to raw tweet data:

1. **Anonymization** — Remove usernames (`@someone`)
2. **Deduplication** — Drop duplicate tweets
3. **Label Remapping** — Convert 3-class to binary
4. **Text Cleaning** — Lowercase, remove punctuation, numbers, hashtags, URLs
5. **Tokenization** — Split into tokens using NLTK
6. **Stopword Removal** — Remove common English stopwords
7. **Lemmatization** — Reduce words to base form using WordNetLemmatizer

---

## 6. Model Training (`src/train.py`)

### Feature Extraction
- **TF-IDF Vectorizer** (unigram) fitted on training data

### Class Balancing
- **SMOTE** (Synthetic Minority Oversampling Technique) applied to handle class imbalance

### Models Trained

#### Support Vector Machine (SVM)
- Kernel: Linear
- Probability: True
- C: 1.0

#### Random Forest (RF)
- n_estimators: 99
- random_state: 42

### Training Command
```powershell
python -m src.train
```

---

## 7. MLflow Experiment Tracking

### Setup
```powershell
mlflow ui
```
UI available at: `http://localhost:5000`

### Tracked per Run
- **Parameters:** Model hyperparameters, test size
- **Metrics:** Accuracy, Precision, Recall, F1 Score (weighted)
- **Artifacts:** Trained model, TF-IDF vectorizer

### Model Registration
- RF model registered in MLflow Model Registry as **`rf_model`**
- Auto-registered on every training run via `registered_model_name="rf_model"` in `log_model()`

---

## 8. Inference (`src/predict.py`)

- Accepts raw text input
- Applies full preprocessing pipeline
- Vectorizes using TF-IDF
- Returns prediction label and confidence score

### Output Format
```json
{
  "input": "original text",
  "cleaned": "preprocessed text",
  "label": "Hate Speech",
  "confidence": 0.87,
  "model": "rf_model (MLflow Registry)"
}
```

---

## 9. Flask REST API (`app.py`)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict hate speech |

### Request Format
```json
{
  "text": "your text here"
}
```

### Run Locally
```powershell
python app.py
```
API available at: `http://localhost:8000`

### Model Loading
- **Locally:** Loads RF model from MLflow registry + vectorizer from MLflow artifact store
- **In Docker:** Loads RF model and vectorizer from `.pkl` files (`USE_PKL=true`)

---

## 10. Streamlit UI (`streamlit_app.py`)

- User enters text in a text area
- Clicks **Predict** button
- Displays:
  - Label (Hate Speech / Not Hate Speech)
  - Confidence score
  - Model name
  - Cleaned text (expandable)

### Run Locally
```powershell
streamlit run streamlit_app.py
```
UI available at: `http://localhost:8501`

---

## 11. Docker Containerization

### Dockerfile
- Base image: `python:3.11-slim`
- Installs all dependencies from `requirements.txt`
- Downloads NLTK resources at build time
- Single image reused across all services

### docker-compose.yml

Three services running from one image (`hate-speech-mlops:latest`):

| Service | Port | Description |
|---------|------|-------------|
| mlflow | 5000 | MLflow tracking server |
| flask-api | 8000 | Flask REST API |
| streamlit | 8501 | Streamlit UI |

### Volumes
- `./mlruns` — MLflow run artifacts
- `./mlflow.db` — MLflow SQLite backend
- `./models` — Trained model pkl files

### Run Full Stack
```powershell
docker-compose up --build
```

---

## 12. GitHub Version Control

- Code hosted at: `https://github.com/ameynagvekar77/hate-speech-mlops`
- `.gitignore` excludes `mlruns/`, `models/*.pkl`, `__pycache__/`, `mlflow.db`
- Every change tracked with `git add`, `git commit`, `git push`

---

## 13. CI/CD Pipeline (GitHub Actions)

File: `.github/workflows/ci.yml`

- Triggers on every push to `main` branch
- Sets up Python 3.11
- Installs dependencies from `requirements.txt`
- Runs `pytest tests/` automatically

### Tests (`tests/test_preprocessing.py`)
- `test_anonymize` — verifies username removal
- `test_clean_text_lowercase` — verifies lowercasing
- `test_clean_text_removes_numbers` — verifies number removal
- `test_clean_text_removes_hashtags` — verifies hashtag removal
- `test_preprocess_text_returns_string` — verifies output type
- `test_preprocess_text_removes_stopwords` — verifies stopword removal

---

## 14. Data Version Control (DVC)

- Initialized with `dvc init`
- Raw dataset tracked with `dvc add data/raw/labeled_data.csv`
- `labeled_data.csv.dvc` pointer file committed to GitHub
- Actual data file excluded from GitHub (too large)

---

## 15. Model Monitoring (Evidently AI)

File: `monitoring.py`

- Splits dataset into **Reference** (70%) and **Current** (30%) data
- Generates HTML report with:
  - **Data Drift Report** — detects feature distribution changes
  - **Classification Performance** — accuracy, precision, recall, F1, confusion matrix

### Run
```powershell
python monitoring.py
```
Opens `monitoring_report.html` in browser.

---

## 16. Real-time Monitoring (Prometheus + Grafana)

### Prometheus
- Scrapes Flask API `/metrics` endpoint every 15 seconds
- Tracks custom metrics:
  - `hate_speech_predictions_total` — count by label
  - `prediction_confidence` — confidence score distribution
  - `flask_http_request_total` — total API requests
  - `flask_http_request_duration_seconds` — API response time

### Grafana
- Visualizes Prometheus metrics in real-time dashboards
- Login: `admin` / `admin`

### Access Points

| Service | URL |
|---------|-----|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## 17. Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| ML Library | scikit-learn |
| Oversampling | imbalanced-learn (SMOTE) |
| NLP | NLTK |
| Experiment Tracking | MLflow |
| API Framework | Flask |
| UI Framework | Streamlit |
| Containerization | Docker + Docker Compose |
| Version Control | Git + GitHub |
| CI/CD | GitHub Actions |
| Data Versioning | DVC |
| Model Monitoring | Evidently AI |
| Real-time Monitoring | Prometheus + Grafana |

---

## 18. How to Run

### Option 1: Locally (3 terminals)
```powershell
# Terminal 1 - MLflow
mlflow ui

# Terminal 2 - Flask API
python app.py

# Terminal 3 - Streamlit
streamlit run streamlit_app.py
```

### Option 2: Docker (1 command)
```powershell
docker-compose up --build
```

### Access Points
| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| Flask API | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## 19. Complete MLOps Pipeline Flow

```
Raw Data
   ↓
DVC (data versioning)
   ↓
Preprocessing (clean, tokenize, lemmatize)
   ↓
Feature Extraction (TF-IDF)
   ↓
Class Balancing (SMOTE)
   ↓
Model Training (SVM + Random Forest)
   ↓
MLflow Tracking (params, metrics, artifacts)
   ↓
Model Registry (rf_model registered)
   ↓
Flask API (serve predictions)
   ↓
Streamlit UI (user interface)
   ↓
Docker (containerized deployment)
   ↓
GitHub + CI/CD (version control + auto testing)
   ↓
Prometheus + Grafana (real-time monitoring)
   ↓
Evidently AI (model performance monitoring)
```
