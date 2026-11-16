
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Find the modeling folder
MODEL_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(MODEL_DIR / "16P.csv", encoding="latin1")

# Clean dataset
df = df.drop("Response Id", axis=1)
df = df.drop_duplicates().reset_index(drop=True)

# Separate features and target
y = df.pop("Personality")
questions = list(df.columns)

# Encode personality labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train SVM model
sv = SVC(C=1, gamma=0.01, kernel="rbf")
sv.fit(x_train, y_train)

# Evaluate model
print("Training accuracy:", sv.score(x_train, y_train))
print("Testing accuracy:", sv.score(x_test, y_test))

# Save all required files inside modeling/
joblib.dump(sv, MODEL_DIR / "svc_model.pkl")
joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
joblib.dump(questions, MODEL_DIR / "questions.pkl")

print("All model files saved successfully!")
