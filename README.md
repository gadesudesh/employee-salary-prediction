# Employee Salary Prediction

A machine learning web application that predicts an employee's annual salary based on personal and professional attributes. Built as a college project using Python, scikit-learn, Flask, and plain HTML/CSS/JavaScript.

---

## Project Overview

This project uses a trained **Random Forest Regressor** to predict the annual salary (in INR) of an employee. The model is served through a **Flask REST API**, and a clean, responsive **frontend** allows users to enter employee details and receive an instant salary estimate — including monthly in-hand, USD equivalent, percentile benchmarking, and +2/+5 year career growth projections.

---

## Problem Statement

Salary transparency and estimation is a critical challenge for both job seekers and hiring organisations. There is no simple, accessible tool that estimates an employee's salary based on their role, experience, and education level without requiring access to confidential HR databases.

This project aims to solve that by training a machine learning model on a publicly available salary dataset and deploying it as a web application.

---

## Objective

- Train and tune a Random Forest Regressor to predict annual salary.
- Build a REST API to serve predictions.
- Provide a clean, user-friendly frontend to interact with the model.
- Demonstrate a complete end-to-end ML project: data → model → API → UI.

---

## Dataset Description

| Property | Details |
|---|---|
| **Source** | Publicly available employee salary dataset |
| **Target variable** | Annual Salary (INR) |
| **Records** | 6,700 |
| **Features** | Age, Gender, Education Level, Job Title, Years of Experience |
| **Unique Job Titles** | 48 |

The dataset contains employee records spanning multiple job titles across various industries, education levels, and experience ranges.

---

## Data Preprocessing

The following preprocessing steps were applied before training:

1. **Missing value removal** — rows with null values were dropped.
2. **Outlier filtering** — unrealistic salary values were removed.
3. **Categorical encoding** — Gender, Education Level, and Job Title were encoded using **OneHotEncoder(handle_unknown='ignore')**.
4. **Numerical scaling** — Age and Years of Experience were standardised using **StandardScaler**.
5. **Pipeline** — preprocessing and the model were combined into a single `sklearn.pipeline.Pipeline` to prevent data leakage.

---

## Exploratory Data Analysis

Key observations from EDA:

- **Years of Experience** and **Job Title** are the strongest salary drivers.
- **Education Level** shows a clear progression: High School < Bachelor's < Master's < PhD.
- **Senior** roles consistently earn significantly more than their Junior equivalents.
- **Age** correlates with salary primarily through its relationship with experience.

---

## Features Used

| Feature | Type | Preprocessing |
|---|---|---|
| Age | Numerical | StandardScaler |
| Gender | Categorical | OneHotEncoder |
| Education Level | Categorical | OneHotEncoder |
| Job Title | Categorical | OneHotEncoder |
| Years of Experience | Numerical | StandardScaler |

**Total features after encoding: ~185** (2 numeric scaled + ~183 OHE columns from 48 job titles, 4 education levels, 2 genders)

---

## Machine Learning Model

Multiple regression models were evaluated during notebook experiments:

| Model | Notes |
|---|---|
| Linear Regression | Baseline model; limited accuracy for non-linear relationships |
| Decision Tree Regressor | Better fit but prone to overfitting |
| **Random Forest Regressor** | ✅ Best performance; robust to overfitting — **deployed** |

Random Forest outperformed the other models due to its ensemble nature — it averages predictions across many decision trees, reducing variance and improving generalisation.

---

## Final Model

**Random Forest Regressor** wrapped in a scikit-learn Pipeline.

| Hyperparameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 20 |
| random_state | 42 |

The `random_state=42` ensures reproducibility of results.

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| n_estimators | 200 |
| max_depth | 20 |
| R² (test set) | **0.9637** (96.37%) |
| MAE | ₹85,170.51 |
| RMSE | ₹1,23,676.25 |
| Test samples | 1,340 |

---

## Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| Years of Experience | ~70.77% |
| Job Title | ~22.34% |
| Education Level | ~5.59% |
| Age | ~1.01% |
| Gender | ~0.28% |

---

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Machine Learning | scikit-learn | 1.9.1 |
| Numerical Computing | NumPy | 2.5.3 |
| Data Manipulation | pandas | 3.0.6 |
| Model Serialisation | joblib | 1.6.0 |
| Backend API | Flask | 3.1.3 |
| CORS Support | flask-cors | 6.0.5 |
| Frontend | HTML5, CSS3, Vanilla JS | — |
| Python | Python | 3.14 |

---

## Project Structure

```
Employee-Salary-Prediction/
│
├── salary_model.pkl        # Trained scikit-learn Pipeline (Random Forest)
├── app.py                  # Flask REST API
├── test_api.py             # Quick API test script
├── metrics.json            # Model performance metrics
├── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── index.html          # Main UI (HTML + embedded JavaScript)
│   └── style.css           # Stylesheet
│
└── venv/                   # Python virtual environment
```

---

## How to Run the Project

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone or download the project

```bash
cd Employee-Salary-Prediction
```

### 2. Create and activate a virtual environment

```bash
# Create venv
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (macOS / Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask API

```bash
python app.py
```

The API will start at: http://127.0.0.1:5000

### 5. Open the frontend

Navigate to http://127.0.0.1:5000 in your browser, or open `frontend/index.html` directly.

---

## How the Prediction Works

```
User fills form
      │
      ▼
Frontend validates input (age, gender, education, job title, experience)
      │
      ▼
JavaScript sends POST /predict with JSON payload
      │
      ▼
Flask backend validates fields (type, range, cross-field rules)
      │
      ▼
pandas DataFrame is built with the 5 required columns
      │
      ▼
sklearn Pipeline runs:
  └─ ColumnTransformer (StandardScaler + OneHotEncoder)
  └─ RandomForestRegressor.predict()
      │
      ▼
Predicted salary (INR) returned as JSON with LPA, USD, percentile, projections
      │
      ▼
Frontend displays full compensation profile
```

---

## Example Prediction

**Input:**

| Field | Value |
|---|---|
| Age | 25 |
| Gender | Male |
| Education Level | Bachelor's |
| Job Title | Software Engineer |
| Years of Experience | 2 |

**Output (example):**

```json
{
  "success": true,
  "predicted_salary": 850000,
  "currency": "INR",
  "salary_lpa": "₹8.50 LPA",
  "model_name": "Random Forest (Tuned)",
  "model_accuracy": "96.37% (R² 0.9637)"
}
```

---

## API Reference

### GET /

Serves the frontend application.

### POST /predict

Predicts annual salary.

**Request body (JSON):**
```json
{
  "age": 25,
  "gender": "Male",
  "education_level": "Bachelor's",
  "job_title": "Software Engineer",
  "years_of_experience": 2
}
```

**Allowed values:**
- gender: "Male", "Female"
- education_level: "High School", "Bachelor's", "Master's", "PhD"
- age: 18 – 80
- years_of_experience: 0 – 60, must be strictly less than age

### GET /api/feature-importance

Returns the Random Forest feature importances aggregated over the 5 parent features.

**Response:**
```json
{
  "success": true,
  "labels": ["Experience", "Job Title", "Education", "Age", "Gender"],
  "values": [70.77, 22.34, 5.59, 1.01, 0.28]
}
```

### GET /api/eda-stats

Returns EDA statistics for the frontend charts.

### GET /api/history

Returns prediction history for the current session.

## Enterprise Features (v2.5 Release)

- **Corporate UI Redesign**: Modern slate & navy theme, enterprise typography (Inter & Plus Jakarta Sans), and professional Font Awesome 6 vector icons (zero emojis).
- **90% Confidence Interval (Ensemble Spread)**: Real-time empirical prediction intervals derived from individual predictions across all 200 decision trees in the Random Forest ensemble.
- **Top Influencing Factors Attribution**: Candidate-specific feature weights and contextual qualitative explanations (Tenure tier, Role market baseline, Academic modifier, Pay equity parity).
- **Real-Time Client & Server Validation**:
  - Age limits: 18 – 80 with live visual feedback indicators.
  - Experience limits: 0 – 60, constrained strictly by `Experience <= Age - 14`.
  - Standardized categoricals: Categorized dropdowns with 48 industry job titles grouped by functional family.
- **In-Memory Caching & Pre-Warmed Engine**: Zero per-request disk reads, pre-warmed inference engine on boot, and static asset cache-control headers (`Cache-Control: public, max-age=86400`).
- **Interactive Sample Profiles**: 1-click test buttons for Entry-Level Software Engineer, Mid-Level Data Analyst, Senior Cloud Architect, and Director of Engineering.
- **Production Scalability & Hosting Optimization**: Clear upgrade paths beyond Render free-tier to eliminate spin-down latency.

---

## Hosting, Reliability & Scalability Architecture

### Overcoming Free-Tier Spin-Down Cold Starts

On free-tier PaaS environments (like Render Free), web services enter sleep mode after 15 minutes of inactivity. The subsequent cold start can take **45 to 60 seconds** to provision the container and load the 67MB model.

To ensure production reliability:
1. **Model Pre-Warming**: `app.py` runs a synthetic warm-up inference during module initialization so the very first client request executes with `< 15ms` latency.
2. **Memory Footprint**: The entire Random Forest ensemble and EDA corpus reside in resident memory (~180MB RAM), operating well within standard 512MB container limits.
3. **HTTP Cache Headers**: Static CSS, JS, and font assets are configured with `Cache-Control: public, max-age=86400`, minimizing network hops and server load.
4. **Recommended Hosting Upgrade**:
   - **Render Starter ($7/mo)**: Keeps instances alive 24/7 with dedicated CPU, completely eliminating free-tier spin-down latency.
   - **Railway / Fly.io**: Fast micro-VM cold starts (< 1.5s) with auto-scaling.
   - **AWS App Runner / GCP Cloud Run**: Set `min-instances = 1` for instantaneous enterprise-grade responsiveness.
   - **Uptime Monitoring**: Configure a free 5-minute health check ping (via UptimeRobot or BetterStack) targeting `https://<your-app>.onrender.com/health` to keep worker processes awake.

### Recommended Gunicorn Production Command

In `Procfile` for production deployment:
```bash
web: gunicorn app:app --workers 4 --threads 2 --worker-class gthread --timeout 120 --keep-alive 5
```

---

## Author

College project — Employee Salary Prediction using Machine Learning

---

*The INR amount shown is predicted directly by the model. The USD display is a frontend approximation using a fixed exchange rate (₹84/USD).*
