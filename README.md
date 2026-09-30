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

---

## Limitations

1. **Fixed job title list** — the model was trained on 48 specific job titles. Unrecognised titles are rejected.
2. **Gender binary** — the dataset only contains Male and Female labels.
3. **INR output** — the model predicts in INR directly.
4. **Static exchange rate** — the USD display uses a hardcoded rate (₹84/USD).
5. **Dataset limitations** — model accuracy depends on training data coverage.
6. **No model retraining** — the model is served as a fixed `.pkl` file.
7. **Version sensitivity** — saved with scikit-learn 1.6.1; a compatibility shim handles 1.9.x loading.

---

## Future Improvements

1. **Add more features** — include industry, company size, and location for more accurate predictions.
2. **Retrain with current scikit-learn** — eliminate the version compatibility shim.
3. **Expand gender options** — update the dataset to support non-binary labels.
4. **Live currency conversion** — fetch the USD/INR rate from a live API.
5. **Model retraining pipeline** — automate retraining as new salary data becomes available.
6. **Confidence intervals** — display a salary range using variance across the 200 trees.
7. **Deploy to cloud** — host the Flask API on Render, Railway, or Google Cloud Run.
8. **Admin dashboard** — monitor prediction counts and input distributions over time.

---

## Author

College project — Employee Salary Prediction using Machine Learning

---

*The INR amount shown is predicted directly by the model. The USD display is a frontend approximation using a fixed exchange rate (₹84/USD).*
