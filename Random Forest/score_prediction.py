import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("student_performance_messy.csv")
print(df.head())

df.info()

print(df.isnull().sum())
print(df.duplicated().sum())

df = df.drop_duplicates()
df["student_name"] = df["student_name"].astype(str).str.strip().str.title()
df["study_hours"] = pd.to_numeric(df["study_hours"].astype(str).str.replace(" hrs", ""), errors="coerce")
df["attendance_percent"] = pd.to_numeric(df["attendance_percent"].astype(str).str.replace("eighty", "80").str.replace("%", ""), errors="coerce")
df["previous_score"] = pd.to_numeric(df["previous_score"].astype(str).str.replace(" / 100", ""), errors="coerce")

df.loc[df["study_hours"] > 24, "study_hours"] = None
df.loc[df["attendance_percent"] > 100, "attendance_percent"] = None
df.loc[df["previous_score"] > 100, "previous_score"] = None
df.loc[df["sleep_hours"] < 0, "sleep_hours"] = None
df = df[df["exam_score"] <= 100]

plt.scatter(df["study_hours"], df["exam_score"])
plt.xlabel("Study hours")
plt.ylabel("Exam score")


features = ["study_hours", "attendance_percent", "previous_score", "sleep_hours"]
X = df[features]
y = df["exam_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance_percent": [90],
    "previous_score": [95],
    "sleep_hours": [8],
})
score = model.predict(new_student)[0]
print("Predicted score:", round(score))
plt.show()