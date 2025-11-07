# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# load
df = pd.read_csv("../data/student_performance.csv")
print("Dataset loaded successfully.")
print(df.head())

# features / target
X = df[["Hours_Studied", "Attendance_Percentage", "Previous_Scores", "Anxiety_Level", "Extracurricular_Activities"]]
y = df["Final_Score"]  # <-- 1D series (fixes warning)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}  |  Testing samples: {len(X_test)}")

# preprocess
numeric_features = ["Hours_Studied", "Attendance_Percentage", "Previous_Scores", "Anxiety_Level"]
categorical_features = ["Extracurricular_Activities"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)  # safer
    ]
)

# models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42)
}

# train + evaluate
for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # works for all sklearn versions
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Performance:")
    print(f"  RMSE (Average Error): {rmse:.2f}")
    print(f"  R^2 (Model Accuracy): {r2:.2f}")

print("\nTraining complete.\n")
