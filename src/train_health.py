# Step 1: Import tools (think of this like opening your toolbox)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Step 2: Load the student data
df = pd.read_csv("../data/Student Mental health.csv")

print("✅ Dataset loaded successfully!")
print(df.head())  # Show first 5 rows
