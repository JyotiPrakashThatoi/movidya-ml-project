
# ---------------------------------------------------------
# This script loads the trained Linear Regression model
# and predicts the final score for new students.
# ---------------------------------------------------------

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# --- STEP 1: Load the dataset again (we'll reuse column info) ---
df = pd.read_csv("../data/student_performance.csv")

# --- STEP 2: Separate features and target ---
X = df[["Hours_Studied", "Attendance_Percentage", "Previous_Scores", "Anxiety_Level", "Extracurricular_Activities"]]
y = df["Final_Score"]

# --- STEP 3: Create the same preprocessor as before ---
numeric_features = ["Hours_Studied", "Attendance_Percentage", "Previous_Scores", "Anxiety_Level"]
categorical_features = ["Extracurricular_Activities"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# --- STEP 4: Rebuild the full pipeline (preprocessor + model) ---
model = LinearRegression()
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# --- STEP 5: Train the pipeline on the entire dataset (for production use) ---
pipe.fit(X, y)

# --- STEP 6: Save the trained model ---
joblib.dump(pipe, "../data/student_model.pkl")
print("Model trained and saved successfully!")

# --- STEP 7: Predict for a new student ---
# Change the numbers below to test different students
new_student = pd.DataFrame({
    "Hours_Studied": [1],
    "Attendance_Percentage": [100],
    "Previous_Scores": [10],
    "Anxiety_Level": [5],
    "Extracurricular_Activities": ["No"]
})

predicted_score = pipe.predict(new_student)[0]
print(f"\nPredicted Final Score: {predicted_score:.2f} / 100")

# Interpret result
if predicted_score >= 50:
    print("Student is likely to PASS.")
else:
    print("Student is at risk of FAILING.")
