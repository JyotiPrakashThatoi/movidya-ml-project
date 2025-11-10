#import necessary libraries
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "../data/Sleep_health_and_lifestyle_dataset.csv"
df = pd.read_csv(file_path)

# Data cleaning
# drop irrelevent columns
df = df.drop(columns=["Person ID", "Blood Pressure"])

# Fill missing values for Sleep Disorder
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

# feautures and target variable
X = df.drop(columns=["Quality of Sleep"])
y = df["Quality of Sleep"]

#indentify categorical and numerical columns
categorical_features = ["Gender", "Occupation", "Sleep Disorder", "BMI Category"]
numerical_features = [col for col in X.columns if col not in categorical_features]

#preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

#Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train and evaluate models
best_model = None
best_score = -float('inf')

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n {name} Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")
    if r2 > best_score:
        best_score = r2
        best_model = pipe

# Save the best model
os.makedirs("../artifacts", exist_ok=True)
joblib.dump(best_model, "../artifacts/sleep_efficiency_model.pkl")
print("\n Best model saved as sleep_efficiency_model.pkl")