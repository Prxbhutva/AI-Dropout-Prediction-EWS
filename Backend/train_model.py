import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# --------------------------
# Example dummy dataset
# Replace this with your real student dataset
# --------------------------
data = {
    "Attendance": [80, 60, 90, 75, 50, 95, 40, 85],
    "CGPA": [8.5, 6.2, 9.0, 7.5, 5.8, 9.2, 4.9, 8.0],
    "Backlogs": [1, 3, 0, 2, 5, 0, 6, 1],
    "Risk": [0, 1, 0, 1, 2, 0, 2, 1]  # Labels: 0=Low, 1=Medium, 2=High
}
df = pd.DataFrame(data)

X = df[["Attendance", "CGPA", "Backlogs"]]
y = df["Risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully as model.pkl and scaler.pkl")
