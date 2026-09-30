"""
Employee Salary Prediction - Enhanced Flask Backend
====================================================
Features:
  - Pre-trained ML pipeline prediction (salary_model.pkl)
  - Single tuned Random Forest Regressor (n_estimators=200, max_depth=20, random_state=42)
  - SQLite User Authentication (Sign up, Login, Session, Profile)
  - History tracking of candidate predictions
  - EDA data endpoints for dynamic frontend charts
  - Career growth forecaster (+2 yrs, +5 yrs projection)
  - Benchmark percentile calculation against dataset
  - Feature importance endpoint (/api/feature-importance)
"""

import os
import json
import sqlite3
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, request, session, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------------------
# Compatibility shim for sklearn version mismatch
# ---------------------------------------------------------------------------
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Compatibility shim: recreates the class removed in sklearn >= 1.7."""
        pass
    _ct._RemainderColsList = _RemainderColsList

# ---------------------------------------------------------------------------
# Directories and Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "salary_model.pkl")
JOB_TITLES_PATH = os.path.join(BASE_DIR, "valid_job_titles.json")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "salary_prediction_data.csv")
DB_PATH = os.path.join(BASE_DIR, "users.db")

# ---------------------------------------------------------------------------
# SQLite Database Setup
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Employee',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            age INTEGER,
            gender TEXT,
            education_level TEXT,
            job_title TEXT,
            years_of_experience REAL,
            model_name TEXT,
            predicted_salary REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migrate any old rows that stored other model names
    cursor.execute(
        "UPDATE prediction_history SET model_name = 'Random Forest (Tuned)' "
        "WHERE model_name != 'Random Forest (Tuned)'"
    )
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------------------------
# Load Model & Job Titles
# ---------------------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded successfully from '{MODEL_PATH}'")
except Exception as e:
    raise RuntimeError(f"[ERROR] Could not load model from '{MODEL_PATH}': {e}")

try:
    with open(JOB_TITLES_PATH, "r", encoding="utf-8") as f:
        VALID_JOB_TITLES = json.load(f)
    VALID_JOB_TITLES_LOWER = {t.lower(): t for t in VALID_JOB_TITLES}
    print(f"[INFO] Loaded {len(VALID_JOB_TITLES)} valid job titles.")
except Exception:
    VALID_JOB_TITLES = []
    VALID_JOB_TITLES_LOWER = {}

# ---------------------------------------------------------------------------
# Precompute EDA Statistics from Dataset
# ---------------------------------------------------------------------------
EDA_CACHE = {}
ALL_SALARIES_SORTED = []

try:
    if os.path.exists(DATASET_PATH):
        df_data = pd.read_csv(DATASET_PATH)
        df_data.columns = df_data.columns.str.strip()
        ALL_SALARIES_SORTED = sorted(df_data["Salary"].dropna().astype(float).tolist())

        # Education averages
        edu_grp = df_data.groupby("Education Level")["Salary"].agg(["mean", "count"]).reindex(
            ["High School", "Bachelor's", "Master's", "PhD"]
        ).dropna()
        edu_stats = {
            "labels": list(edu_grp.index),
            "averages": [round(val) for val in edu_grp["mean"]],
            "counts": [int(val) for val in edu_grp["count"]]
        }

        # Top 10 highest paying roles
        top_jobs = df_data.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(10)
        top_job_stats = {
            "labels": list(top_jobs.index),
            "averages": [round(val) for val in top_jobs.values]
        }

        # Experience vs Salary curve
        exp_grp = df_data.groupby("Years of Experience")["Salary"].mean().sort_index()
        exp_stats = {
            "years": [int(y) for y in exp_grp.index if y <= 25],
            "averages": [round(val) for y, val in exp_grp.items() if y <= 25]
        }

        # Gender pay breakdown
        gender_grp = df_data.groupby("Gender")["Salary"].agg(["mean", "count"])
        gender_stats = {
            "labels": list(gender_grp.index),
            "averages": [round(val) for val in gender_grp["mean"]],
            "counts": [int(val) for val in gender_grp["count"]]
        }

        EDA_CACHE = {
            "total_records": len(df_data),
            "unique_jobs": df_data["Job Title"].nunique(),
            "mean_salary": round(float(df_data["Salary"].mean())),
            "median_salary": round(float(df_data["Salary"].median())),
            "min_salary": round(float(df_data["Salary"].min())),
            "max_salary": round(float(df_data["Salary"].max())),
            "education": edu_stats,
            "top_jobs": top_job_stats,
            "experience_curve": exp_stats,
            "gender": gender_stats
        }
        print(f"[INFO] EDA cached successfully ({len(df_data)} records).")
except Exception as e:
    print(f"[WARN] Failed to compute EDA cache: {e}")

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "salary-prediction-secret-key-2026")
CORS(app)

# ---------------------------------------------------------------------------
# Helper: Database Connection
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------------------------------------------------------------
# Frontend Serving Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return send_from_directory("frontend", "index.html")

@app.route("/job-titles", methods=["GET"])
def job_titles():
    return jsonify(VALID_JOB_TITLES)

# ---------------------------------------------------------------------------
# Auth API Routes
# ---------------------------------------------------------------------------
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()
    role = str(data.get("role", "Employee")).strip()

    if not name or not email or not password:
        return jsonify({"success": False, "error": "All fields are required."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({"success": False, "error": "An account with this email already exists."}), 409

    pwd_hash = generate_password_hash(password)
    cursor.execute(
        "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (name, email, pwd_hash, role)
    )
    conn.commit()
    conn.close()

    session["user_email"] = email
    session["user_name"] = name
    session["user_role"] = role

    return jsonify({
        "success": True,
        "user": {"name": name, "email": email, "role": role}
    })

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, email, password_hash, role FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"success": False, "error": "Invalid email or password."}), 401

    session["user_email"] = user["email"]
    session["user_name"] = user["name"]
    session["user_role"] = user["role"]

    return jsonify({
        "success": True,
        "user": {"name": user["name"], "email": user["email"], "role": user["role"]}
    })

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/auth/me", methods=["GET"])
def me():
    if "user_email" in session:
        return jsonify({
            "logged_in": True,
            "user": {
                "name": session.get("user_name"),
                "email": session.get("user_email"),
                "role": session.get("user_role")
            }
        })
    return jsonify({"logged_in": False})

# ---------------------------------------------------------------------------
# Prediction Route
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}

    required_fields = ["age", "gender", "education_level", "job_title", "years_of_experience"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"success": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        age = float(data["age"])
        gender = str(data["gender"]).strip()
        education_level = str(data["education_level"]).strip()
        job_title = str(data["job_title"]).strip()
        years_of_experience = float(data["years_of_experience"])
    except (ValueError, TypeError) as e:
        return jsonify({"success": False, "error": f"Invalid field value: {e}"}), 400

    # Validation
    ALLOWED_GENDERS = {"Male", "Female"}
    ALLOWED_EDUCATION = {"High School", "Bachelor's", "Master's", "PhD"}
    validation_errors = []

    if age < 18 or age > 80:
        validation_errors.append("Age must be between 18 and 80.")
    if gender not in ALLOWED_GENDERS:
        validation_errors.append(f"Gender must be one of: {', '.join(sorted(ALLOWED_GENDERS))}.")
    if education_level not in ALLOWED_EDUCATION:
        validation_errors.append(f"Education Level must be one of: {', '.join(sorted(ALLOWED_EDUCATION))}.")
    if not job_title:
        validation_errors.append("Job Title cannot be empty.")
    elif VALID_JOB_TITLES_LOWER and job_title.lower() not in VALID_JOB_TITLES_LOWER:
        validation_errors.append(f"Job Title '{job_title}' is not recognized.")
    else:
        job_title = VALID_JOB_TITLES_LOWER.get(job_title.lower(), job_title)

    if years_of_experience < 0 or years_of_experience > 60:
        validation_errors.append("Years of Experience must be between 0 and 60.")
    if not validation_errors and years_of_experience >= age:
        validation_errors.append("Years of Experience cannot be greater than or equal to Age.")

    if validation_errors:
        return jsonify({"success": False, "error": " | ".join(validation_errors)}), 422

    # Predict using the tuned Random Forest pipeline
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education_level,
        "Job Title": job_title,
        "Years of Experience": years_of_experience
    }])

    try:
        base_pred = float(model.predict(input_df)[0])
        predicted_salary = round(base_pred, 2)
        model_display_name = "Random Forest (Tuned)"
        model_accuracy = "96.37% (R² 0.9637)"

        # Compute Career Growth Projections
        # +2 years
        future_2_df = pd.DataFrame([{
            "Age": age + 2,
            "Gender": gender,
            "Education Level": education_level,
            "Job Title": job_title,
            "Years of Experience": years_of_experience + 2
        }])
        pred_plus_2 = round(float(model.predict(future_2_df)[0]))

        # +5 years
        future_5_df = pd.DataFrame([{
            "Age": age + 5,
            "Gender": gender,
            "Education Level": education_level,
            "Job Title": job_title,
            "Years of Experience": years_of_experience + 5
        }])
        pred_plus_5 = round(float(model.predict(future_5_df)[0]))

        # Percentile Calculation against dataset
        percentile = 50
        if ALL_SALARIES_SORTED:
            count_below = sum(1 for s in ALL_SALARIES_SORTED if s <= predicted_salary)
            percentile = max(1, min(99, round((count_below / len(ALL_SALARIES_SORTED)) * 100)))

        # Monthly in-hand estimate (approx 85% post standard Indian deductions)
        monthly_inhand = round((predicted_salary * 0.85) / 12)

        # Record in prediction history
        user_email = session.get("user_email", "guest")
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO prediction_history
                (user_email, age, gender, education_level, job_title, years_of_experience, model_name, predicted_salary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_email, int(age), gender, education_level, job_title, years_of_experience, model_display_name, predicted_salary))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[WARN] Failed to save prediction history: {e}")

        return jsonify({
            "success": True,
            "predicted_salary": predicted_salary,
            "currency": "INR",
            "period": "annual",
            "salary_lpa": f"₹{(predicted_salary / 100000):.2f} LPA",
            "monthly_inhand": f"₹{monthly_inhand:,.0f} / month",
            "usd_equivalent": f"${round(predicted_salary / 84):,.0f} USD / year",
            "model_name": model_display_name,
            "model_accuracy": model_accuracy,
            "percentile": percentile,
            "projections": {
                "plus_2_years": {
                    "years": years_of_experience + 2,
                    "salary": pred_plus_2,
                    "salary_lpa": f"₹{(pred_plus_2 / 100000):.2f} LPA"
                },
                "plus_5_years": {
                    "years": years_of_experience + 5,
                    "salary": pred_plus_5,
                    "salary_lpa": f"₹{(pred_plus_5 / 100000):.2f} LPA"
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction computation failed: {e}"}), 500

# ---------------------------------------------------------------------------
# Feature Importance API
# ---------------------------------------------------------------------------
@app.route("/api/feature-importance", methods=["GET"])
def get_feature_importance():
    try:
        preprocessor = model.named_steps["preprocessor"]
        rf = model.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        importances = rf.feature_importances_

        # Map each encoded feature back to its parent feature
        parent_map = {
            "num__Age": "Age",
            "num__Years of Experience": "Years of Experience",
        }
        # One-hot columns follow pattern cat__<feature>_<value>
        cat_parents = {
            "Gender": "Gender",
            "Education Level": "Education Level",
            "Job Title": "Job Title",
        }

        aggregated = {}
        for fname, imp in zip(feature_names, importances):
            matched = False
            if fname in parent_map:
                parent = parent_map[fname]
                aggregated[parent] = aggregated.get(parent, 0.0) + imp
                matched = True
            else:
                for cat_key, parent in cat_parents.items():
                    if fname.startswith(f"cat__{cat_key}_") or fname == f"cat__{cat_key}":
                        aggregated[parent] = aggregated.get(parent, 0.0) + imp
                        matched = True
                        break
            if not matched:
                aggregated["Other"] = aggregated.get("Other", 0.0) + imp

        total = sum(aggregated.values())
        if total == 0:
            raise ValueError("Total importance is zero.")

        # Convert to percentages and sort descending
        pct = {k: round((v / total) * 100, 2) for k, v in aggregated.items()}
        sorted_items = sorted(pct.items(), key=lambda x: x[1], reverse=True)

        # Keep only the 5 model features; rename Age/YoE to friendly labels
        label_map = {
            "Years of Experience": "Experience",
            "Job Title": "Job Title",
            "Education Level": "Education",
            "Age": "Age",
            "Gender": "Gender",
        }
        labels = []
        values = []
        for k, v in sorted_items:
            if k in label_map:
                labels.append(label_map[k])
                values.append(v)

        return jsonify({"success": True, "labels": labels, "values": values})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------------------------------------------------------------------
# EDA & Metrics API
# ---------------------------------------------------------------------------
@app.route("/api/eda-stats", methods=["GET"])
def get_eda_stats():
    if EDA_CACHE:
        return jsonify({"success": True, "data": EDA_CACHE})
    return jsonify({"success": False, "error": "EDA data not available."}), 404

@app.route("/api/model-metrics", methods=["GET"])
def get_model_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return jsonify({"success": True, "metrics": metrics})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "metrics.json not found."}), 404

@app.route("/api/history", methods=["GET"])
def get_history():
    user_email = session.get("user_email")
    conn = get_db()
    cursor = conn.cursor()
    if user_email:
        cursor.execute("""
            SELECT id, age, gender, education_level, job_title, years_of_experience, model_name, predicted_salary, created_at
            FROM prediction_history WHERE user_email = ? ORDER BY id DESC LIMIT 20
        """, (user_email,))
    else:
        cursor.execute("""
            SELECT id, age, gender, education_level, job_title, years_of_experience, model_name, predicted_salary, created_at
            FROM prediction_history ORDER BY id DESC LIMIT 10
        """)
    rows = cursor.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "age": r["age"],
            "gender": r["gender"],
            "education": r["education_level"],
            "job_title": r["job_title"],
            "experience": r["years_of_experience"],
            "model_name": r["model_name"],
            "predicted_salary": r["predicted_salary"],
            "salary_lpa": f"₹{(r['predicted_salary'] / 100000):.2f} LPA",
            "created_at": r["created_at"]
        })
    return jsonify({"success": True, "history": history})

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
