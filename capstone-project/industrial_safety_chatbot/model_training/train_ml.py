# train_ml.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import json

# Load cleaned data
df = pd.read_csv("cleaned_dataset.csv")
X = df["clean_text"]
y = df["Accident Level"]

# Vectorize
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "NaiveBayes": MultinomialNB(),
}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics[name] = {"accuracy": report["accuracy"], "macro avg": report["macro avg"]}

    # Save model
    joblib.dump(model, f"models/model_{name.lower()}.pkl")

# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

# Save metrics
with open("metrics/ml_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ ML models trained and metrics saved.")
