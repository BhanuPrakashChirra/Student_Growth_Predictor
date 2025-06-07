# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For nicer plots
sns.set(style='whitegrid')

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\bhanu\Desktop\Student\student_performance_data.csv")  # Adjust the path if needed
print(df.head())

# Step 3: Dataset Overview and Missing Values Check
print("Dataset shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values in each column:\n", df.isnull().sum())

# Step 4: Descriptive Statistics for Numerical Features
print(df.describe())

# Step 5: Visualize Target Variable Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='AcademicPerformanceStatus', palette='Set2')
plt.title("Distribution of Academic Performance (Pass/Fail)")
plt.xlabel("Performance")
plt.ylabel("Count")
plt.show()

# Step 6: Visualize Categorical Features Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(ax=axes[0], data=df, x='Gender', palette='cool')
axes[0].set_title("Gender Distribution")

sns.countplot(ax=axes[1], data=df, x='SocioeconomicStatus', palette='coolwarm')
axes[1].set_title("Socioeconomic Status Distribution")

sns.countplot(ax=axes[2], data=df, x='ClassParticipation', palette='cubehelix')
axes[2].set_title("Class Participation Levels")

plt.tight_layout()
plt.show()

# Step 7: Visualize Numerical Features Distributions
num_cols = ['Age', 'Grades', 'Attendance', 'TimeSpentOnHomework']

df[num_cols].hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Histogram of Numerical Features", fontsize=16)
plt.show()

# Step 8: Correlation Matrix for Numerical Features
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Step 9: Relationship of Features with Target Variable
plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
sns.boxplot(data=df, x='AcademicPerformanceStatus', y='Grades', palette='pastel')
plt.title("Grades vs Academic Performance")

plt.subplot(3,1,2)
sns.boxplot(data=df, x='AcademicPerformanceStatus', y='Attendance', palette='Set3')
plt.title("Attendance vs Academic Performance")

plt.subplot(3,1,3)
sns.boxplot(data=df, x='AcademicPerformanceStatus', y='TimeSpentOnHomework', palette='Accent')
plt.title("Time Spent on Homework vs Academic Performance")

plt.tight_layout()
plt.show()

# Step 10: Pairplot of Key Numerical Features Colored by Performance
sns.pairplot(df, hue='AcademicPerformanceStatus', vars=['Grades', 'Attendance', 'TimeSpentOnHomework'], palette='husl')
plt.suptitle("Pairplot of Key Numerical Features", y=1.02)
plt.show()
