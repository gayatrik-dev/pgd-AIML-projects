# train_lstm.py
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("cleaned_dataset.csv")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["Accident Level"])
y_cat = to_categorical(y)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["clean_text"])
X_seq = tokenizer.texts_to_sequences(df["clean_text"])
X_pad = pad_sequences(X_seq, maxlen=100)

X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_cat, test_size=0.2, random_state=42
)

# LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(6, activation="softmax"))  # 6 classes

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate
y_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

report = classification_report(y_test_labels, y_pred_labels, output_dict=True)

metrics = {"accuracy": report["accuracy"], "macro avg": report["macro avg"]}

# Save model + tokenizer + metrics
model.save("models/lstm_model.h5")
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("metrics/lstm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ LSTM model trained and metrics saved.")
