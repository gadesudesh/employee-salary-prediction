import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "age": 25,
    "gender": "Male",
    "education_level": "Bachelor's",
    "job_title": "Software Engineer",
    "years_of_experience": 2
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
res_json = response.json()
print("Success:", res_json.get("success"))
print("Model:", res_json.get("model_name"))
print("Predicted Salary (INR):", res_json.get("predicted_salary"))
print("LPA:", res_json.get("salary_lpa", "").encode("ascii", "replace").decode("ascii"))
print("Monthly In-hand:", res_json.get("monthly_inhand", "").encode("ascii", "replace").decode("ascii"))
print("Percentile:", res_json.get("percentile"))
ci = res_json.get("confidence_interval", {})
print("Confidence Interval (Range):", ci.get("range_text", "").encode("ascii", "replace").decode("ascii"))
print("Confidence Interval (Std Dev):", ci.get("std_dev_lpa", "").encode("ascii", "replace").decode("ascii"))
factors = res_json.get("influencing_factors", [])
print(f"Top Influencing Factors ({len(factors)}):")
for f in factors:
    print(f"  - {f.get('factor')}: {f.get('value')} ({f.get('weight_pct')}%) [{f.get('impact')}]")

# Verify feature importance endpoint
fi_response = requests.get("http://127.0.0.1:5000/api/feature-importance")
fi_json = fi_response.json()
print("\n--- Feature Importance ---")
print("Success:", fi_json.get("success"))
if fi_json.get("success"):
    total = sum(fi_json.get("values", []))
    print("Labels:", fi_json.get("labels"))
    print("Values:", fi_json.get("values"))
    print("Total (should be ~100):", round(total, 2))
