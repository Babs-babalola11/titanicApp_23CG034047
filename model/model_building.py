# ==============================
# Titanic Survival Prediction
# ==============================

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ------------------------------
# Step 2: Load dataset
# ------------------------------
df = pd.read_csv("titanic.csv")  # Replace with your dataset path
print("Dataset Loaded")
print(df.head())
print(df.info())

# ------------------------------
# Step 3: Data Preprocessing
# ------------------------------

# 3a: Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Drop column with too many missing values

# 3b: Feature Selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# 3c: Encode categorical variables
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# 3d: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Step 4: Split dataset
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# Step 5: Train Model
# ------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Step 6: Evaluate Model
# ------------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------
# Step 7: Save Model & Scaler
# ------------------------------
joblib.dump(model, "titanic_survival_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and scaler saved successfully!")

# ------------------------------
# Step 8: Reload Model for Prediction
# ------------------------------
loaded_model = joblib.load("titanic_survival_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# Example prediction for first 5 test samples
sample = X_test[:5]
sample_pred = loaded_model.predict(sample)
print("\nSample Predictions (0=Dead, 1=Survived):", sample_pred)
