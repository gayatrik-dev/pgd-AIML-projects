# train_fnn.py
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import joblib

# Load data
df = pd.read_csv("cleaned_dataset.csv")
X = df["clean_text"]
y = df["Accident Level"]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X).toarray()

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_cat, test_size=0.2, random_state=42
)

# Define FNN model
model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(X_vec.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(6, activation="softmax"))  # 6 accident levels

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
)

# Evaluate
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

report = classification_report(y_true, y_pred_class, output_dict=True)
metrics = {"accuracy": report["accuracy"], "macro avg": report["macro avg"]}

# Save model and vectorizer
model.save("models/fnn_model.h5")

joblib.dump(vectorizer, "models/fnn_vectorizer.pkl")

# Save metrics
with open("metrics/fnn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ FNN model trained and metrics saved.")
