# src/feature_importance_sleep.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load dataset & model
df = pd.read_csv("../data/Sleep_health_and_lifestyle_dataset.csv")
df = df.drop(columns=["Person ID", "Blood Pressure"])
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

model = joblib.load("../artifacts/sleep_efficiency_model.pkl")

# 2) Recover fitted column groups from the pipeline itself
pre = model.named_steps["preprocessor"]

# ColumnTransformer stores tuples: (name, transformer, columns)
num_cols = list(pre.transformers_[0][2])  # numeric column names used during fit
cat_cols = list(pre.transformers_[1][2])  # categorical column names used during fit
ohe = pre.transformers_[1][1]             # the fitted OneHotEncoder

# 3) Build categorical feature names from categories_ (version-proof)
cat_feature_names = []
for col, cats in zip(cat_cols, ohe.categories_):
    for c in cats:
        cat_feature_names.append(f"{col}={c}")

all_feature_names = num_cols + cat_feature_names

# 4) Get importances / coefficients from the final estimator
est = model.named_steps["model"]

if hasattr(est, "feature_importances_"):
    importances = est.feature_importances_
    scores = importances
    kind = "Importance"
elif hasattr(est, "coef_"):
    # LinearRegression: use absolute coefficient magnitude for ranking
    coefs = np.ravel(est.coef_)
    scores = np.abs(coefs)
    kind = "Coefficient |abs|"
else:
    raise RuntimeError("The final estimator has neither feature_importances_ nor coef_.")

# 5) Create a sorted table
fi = (pd.DataFrame({"Feature": all_feature_names, "Score": scores})
        .sort_values("Score", ascending=False)
        .reset_index(drop=True))

print("\nTop 15 features:\n", fi.head(15))

# 6) Plot top 15
topk = 10
plt.figure(figsize=(9,5))
plt.barh(fi.loc[:topk-1, "Feature"], fi.loc[:topk-1, "Score"])
plt.gca().invert_yaxis()
plt.title(f"Top {topk} Features ({kind})")
plt.xlabel(kind)
plt.tight_layout()
plt.show()
