# Step 1: Import Required Libraries
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

# Step 2: Load Preprocessed Data
df = pd.read_csv(r"C:\Users\bhanu\Desktop\Student\student_performance_data.csv")
X = df.drop(columns=['StudentID', 'AcademicPerformanceStatus'])
y = df['AcademicPerformanceStatus']

# Step 3: Encode Target Variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)  # 'Pass' -> 1, 'Fail' -> 0 (depends on original data)

# Step 4: Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 5: Scale Numerical Features
num_features = ['Age', 'Grades', 'Attendance', 'TimeSpentOnHomework']
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Step 6: Encode Categorical Features
categorical_cols = ['Gender', 'SocioeconomicStatus', 'ClassParticipation']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Step 7: Train Logistic Regression (Baseline)
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)

# Step 8: Train XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Confusion Matrix for XGBoost
cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.show()

# Step 9: Hyperparameter Tuning (XGBoost)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# Step 10: Train Tuned XGBoost Model
best_params = grid_search.best_params_
xgb_tuned = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **best_params)
xgb_tuned.fit(X_train, y_train)
y_pred_tuned = xgb_tuned.predict(X_test)
print(f"Tuned XGBoost Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

# Final Evaluation
print("Classification Report for Tuned XGBoost:\n", classification_report(y_test, y_pred_tuned))
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Tuned XGBoost')
plt.show()

# Step 11: Save Final Model and Scaler
joblib.dump(xgb_tuned, "../api/xgb_student_model.pkl")
joblib.dump(scaler, "../api/scaler.pkl")
print("Tuned model and scaler saved successfully.")
