# save_model.py — trains on the existing student_performance.csv inside /data folder
# and saves the trained model pipeline + numeric feature ranges.

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------
# 1️⃣ Define paths relative to your project folder
# ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent           # current project folder
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"  # our existing CSV inside /data

# ----------------------------------------------------------------
# 2️⃣ Load the dataset
# ----------------------------------------------------------------
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ----------------------------------------------------------------
# 3️⃣ Define columns
# ----------------------------------------------------------------
numeric_features = ["Hours_Studied", "Attendance_Percentage", "Previous_Scores", "Anxiety_Level"]
categorical_features = ["Extracurricular_Activities"]
target = "Final_Score"

X = df[numeric_features + categorical_features]
y = df[target]

# ----------------------------------------------------------------
# 4️⃣ Build the preprocessing + model pipeline
# ----------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = LinearRegression()
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# ----------------------------------------------------------------
# 5️⃣ Train the model
# ----------------------------------------------------------------
pipe.fit(X, y)
print("✅ Model trained successfully.")

# ----------------------------------------------------------------
# 6️⃣ Save model and feature ranges
# ----------------------------------------------------------------
MODEL_PATH = BASE_DIR / "student_model.joblib"
RANGE_PATH = BASE_DIR / "feature_ranges.joblib"

# Save trained pipeline
joblib.dump(pipe, MODEL_PATH)

# Save numeric feature ranges for later input validation
feature_ranges = {col: [float(df[col].min()), float(df[col].max())] for col in numeric_features}
joblib.dump(feature_ranges, RANGE_PATH)

print(f"✅ Saved model to: {MODEL_PATH}")
print(f"✅ Saved feature ranges to: {RANGE_PATH}")
