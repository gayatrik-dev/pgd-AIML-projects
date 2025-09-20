import pandas as pd
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_excel("Industrial_safety_and_health_database.xlsx")
df = df.dropna(subset=["Description", "Accident Level"])
df["Description"] = df["Description"].str.lower().str.strip()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words
    ]
    return " ".join(tokens)


df["clean_text"] = df["Description"].apply(preprocess)
df.to_csv("cleaned_dataset.csv", index=False)
