import joblib

# ================================
# Load Your Models
# ================================
sql_vectorizer = joblib.load("SQLInjection_vectorizer.pkl")
sql_model = joblib.load("SQLInjection_Model.pkl")

xss_vectorizer = joblib.load("xss_vectorizer.pkl")
xss_model = joblib.load("xss_sqli_model.pkl")

# ================================
# Predict Function
# ================================
def test_model(model, vectorizer, text):
    """Return prediction and probability for a given model."""
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    # Some models support predict_proba, some don't
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0].max()
    else:
        prob = "N/A (model does not support predict_proba)"

    return pred, prob


# ================================
# Main Test Loop
# ================================
while True:
    user_input = input("\nEnter payload to test (or 'q' to quit): ")

    if user_input.lower() == "q":
        break

    # ---------- Test SQL Injection Model ----------
    sql_pred, sql_prob = test_model(sql_model, sql_vectorizer, user_input)

    # ---------- Test XSS/SQLi Combined Model ----------
    xss_pred, xss_prob = test_model(xss_model, xss_vectorizer, user_input)

    print("\n========================")
    print("🔍 INPUT PAYLOAD:", user_input)
    print("========================")

    print("\n📌 SQL Injection Model")
    print("Prediction:", sql_pred)
    print("Confidence:", sql_prob)

    print("\n📌 XSS / SQL Injection Combined Model")
    print("Prediction:", xss_pred)
    print("Confidence:", xss_prob)

    print("\n----------------------------------------")
