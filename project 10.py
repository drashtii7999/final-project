
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset

import os

folder = os.path.dirname(os.path.abspath(__file__))
print("Files in folder:", os.listdir(folder))

path = os.path.join(folder, "titanic - Copy.csv")
print("CSV exists:", os.path.exists(path))

df = pd.read_csv(path)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 2. Dataset Information
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

# 3. Clean Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

print("\nMissing values cleaned!")

# 4. Survival Count
sns.countplot(data=df, x="Survived")
plt.title("Titanic Survival Count")
plt.show()

# 5. Survival by Gender
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.show()

# 6. Survival by Passenger Class
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.show()

# 7. Passenger Age Distribution
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Passenger Age Distribution")
plt.show()

# 8. Survival Rate
print("\nSurvival Rate:")
print(df["Survived"].mean() * 100, "%")

# 9. Feature Correlation
sns.heatmap(
    df[["Survived", "Pclass", "Age", "SibSp",
        "Parch", "Fare"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")
plt.show()

# 10. Save Cleaned Dataset
df.to_csv("cleaned_titanic.csv", index=False)
print("Cleaned dataset saved!")


