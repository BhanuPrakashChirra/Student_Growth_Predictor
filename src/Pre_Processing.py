# student_performance_pipeline.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import joblib

# Step 1: Load Dataset
df = pd.read_csv(r"C:\Users\bhanu\Desktop\Student\student_performance_data.csv")

# Step 2: Check Dataset Shape and Info
print("Dataset shape:", df.shape)
df.info()

# Step 3: Check for Missing Values
print("\nMissing values in each column:\n", df.isnull().sum())

# Step 4: Summary Statistics of Numerical Features
print("\nSummary statistics:\n", df.describe())

# Step 5: Encode Categorical Variables
categorical_cols = ['Gender', 'SocioeconomicStatus', 'ClassParticipation', 'AcademicPerformanceStatus']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 6: Drop Unnecessary Columns
df.drop(columns=['StudentID'], inplace=True)

# Step 7: Split Features and Target Variable
X = df.drop('AcademicPerformanceStatus', axis=1)
y = df['AcademicPerformanceStatus']

# Step 8: Split Dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 9: Scale Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Logistic Regression Model
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))

# Step 11: XGBoost Model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)

print("\n--- XGBoost Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Step 12: Feature Importance (XGBoost)
plt.figure(figsize=(10, 6))
plot_importance(xgb)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# Step 13: Save Encoders and Scaler
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(logreg, 'logistic_regression_model.pkl')
joblib.dump(xgb, 'xgboost_model.pkl')

# Optional: Check Class Distribution
print("\nClass distribution:\n", y.value_counts(normalize=True))
