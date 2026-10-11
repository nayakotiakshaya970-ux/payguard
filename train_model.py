import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
df = pd.read_csv("data/invoices.csv")

# 2. Convert the industry text into numbers (models only understand numbers)
le_industry = LabelEncoder()
df["industry_encoded"] = le_industry.fit_transform(df["industry"])

# 3. Choose inputs (X) and the answer to predict (y)
features = [
    "client_age_months",
    "past_late_payments",
    "invoice_amount",
    "payment_terms_days",
    "industry_encoded",
    "client_rating",
]
X = df[features]
y = df["paid_late"]

# 4. Split: 80% for learning, 20% hidden for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train the model
model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# 6. Test it on invoices it has never seen
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 Score:", round(f1_score(y_test, y_pred), 3))

# 7. Which factors matter most?
importances = pd.Series(model.feature_importances_, index=features)
print("\nTop risk factors:\n", importances.sort_values(ascending=False).round(3))

# 8. Save everything the dashboard will need
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/payguard_model.pkl")
joblib.dump(le_industry, "models/industry_encoder.pkl")
joblib.dump(features, "models/feature_names.pkl")
print("\nModel saved to models/")