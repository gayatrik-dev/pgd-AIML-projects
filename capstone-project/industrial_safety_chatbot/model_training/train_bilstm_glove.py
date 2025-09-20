# train_bilstm_glove.py

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

texts = df["clean_text"].values
labels = df["Accident Level"].values

# Encode labels
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
y_cat = to_categorical(labels_enc)

# Tokenize
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

max_len = 100
X = pad_sequences(sequences, maxlen=max_len)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# 🔹 Load GloVe embeddings
print("Loading GloVe embeddings...")
embeddings_index = {}
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

# Prepare embedding matrix
embedding_dim = 100
num_words = min(10000, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i >= 10000:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("✅ GloVe embedding matrix ready.")

# 🔹 Build BiLSTM model
model = Sequential()
model.add(
    Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False,
    )
)  # Freeze GloVe embeddings
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(6, activation="softmax"))  # 6 accident levels

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 🔹 Train
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
)

# 🔹 Evaluate
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

report = classification_report(y_true, y_pred_class, output_dict=True)

metrics = {"accuracy": report["accuracy"], "macro avg": report["macro avg"]}

# 🔹 Save everything
model.save("models/bilstm_glove_model.h5")
with open("models/tokenizer_bilstm.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("metrics/bilstm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ BiLSTM with GloVe model trained and saved.")
