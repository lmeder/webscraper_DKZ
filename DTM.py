# --- Imports ---
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import random
import numpy as np
import tomotopy as tp
from collections import Counter

# ---------- 0️⃣ NLTK STOPWORDS HERUNTERLADEN ----------
nltk.download("stopwords")
nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])

# ---------- 1️⃣ CSV LADEN ----------
df = pd.read_csv(
    "filtered_labels_output.csv",
    sep=";",
    quotechar='"',
    on_bad_lines="skip",
    engine="python",
)
df = df[["text", "label", "class"]].dropna()

# ---------- 2️⃣ OCR-NORMALISIERUNG ----------
def normalize_ocr(text):
    text = str(text)
    text = text.replace("ſ", "s")
    replacements = {
        "ﬁ": "fi", "ﬂ": "fl",
        "ä": "ae", "ö": "oe", "ü": "ue",
        "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
        "ß": "ss",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(normalize_ocr)

# ---------- 3️⃣ JAHR AUS LABEL ----------
def extract_year(label):
    m = re.search(r"(18|19|20)\d{2}", str(label))
    if m:
        return int(m.group())
    return None

df["year"] = df["label"].apply(extract_year)
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# --- 4️⃣ Tokenisierung + Lemmatization mit spaCy ---
import spacy
from tqdm import tqdm

# Lade das kleine deutsche Modell
nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])  # nur Tokenizer + Lemmatizer

# Stopwords
stop_words = set(stopwords.words("german"))
custom_stopwords = {
    "dass","daß","der","die","das","und","mit","von","zu","den","des",
    "im","für","fuer","auf","ist","nicht",
    "aber","als","auch","aus","bei","bis",
    "worden","wurde","wurden","fuer","ueber","wegen",
    "waere","sei","wuerde","mehr","immer","erst","schon",
    "grosse","grossen","kleinen","ganz","teil","zeit","jahre","ueber","waehrend", 
    "koennen","kommen","muessen",
    "the","ver","ven"
}
stop_words.update(custom_stopwords)

def spacy_tokenize_lemmatize(docs, batch_size=5000, n_process=1):
    """
    docs: Liste von Strings
    batch_size: Anzahl Dokumente pro Batch
    n_process: Anzahl Prozesse (nur >1 bei spaCy >=3.0)
    """
    all_tokens = []
    for doc in tqdm(nlp.pipe(docs, batch_size=batch_size, n_process=n_process), total=len(docs)):
        tokens = [token.lemma_.lower() 
                  for token in doc 
                  if not token.is_punct and not token.is_space 
                  and token.lemma_.lower() not in stop_words 
                  and len(token.lemma_) > 2]
        all_tokens.append(tokens)
    return all_tokens

# wende die Funktion auf die clean_text-Spalte an
df["tokens"] = spacy_tokenize_lemmatize(df["clean_text"].tolist(), batch_size=5000, n_process=1)

# --- 5️⃣ Kurze Dokumente rausfiltern ---
min_tokens = 3
df_filtered = df[df["tokens"].apply(len) >= min_tokens].copy()

# --- 5.1️⃣ Hapax-Legomena entfernen ---
from collections import Counter
all_tokens = [token for doc in df_filtered["tokens"] for token in doc]
freq = Counter(all_tokens)
hapax = {w for w, c in freq.items() if c == 1}

df_filtered["tokens"] = df_filtered["tokens"].apply(
    lambda doc: [w for w in doc if w not in hapax]
)

# --- 5.2️⃣ Dokumente erneut prüfen ---
df_filtered = df_filtered[df_filtered["tokens"].apply(len) >= min_tokens].copy()

# Optional: Testlauf
df_sample = df_filtered.head(150_000)

print("Ursprüngliche Dokumente:", len(df))
print("Nach Filterung:", len(df_filtered))
print("Testlauf:", len(df_sample))
print("Vokabulargröße nach Hapax-Filter:", len(set([token for doc in df_filtered["tokens"] for token in doc])))


# Optional: Sample für schnellere Tests
df_sample = df_filtered.head(150_000)
print("Testlauf:", len(df_sample))

# ---------- 6️⃣ DTM SETUP ----------
years = sorted(df_filtered["year"].unique())
year_to_tp = {year: idx for idx, year in enumerate(years)}

random.seed(42)
np.random.seed(42)

dtm = tp.DTModel(
    k=15,
    t=len(years),
    min_cf=10,
    min_df=3,
    tw=tp.TermWeight.IDF
)

# ---------- 7️⃣ DOKUMENTE HINZUFÜGEN ----------
for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    dtm.add_doc(row["tokens"], timepoint=year_to_tp[row["year"]])

dtm.burn_in = 100

# ---------- 8️⃣ TRAINING ----------
print("Starte Training...")
for i in range(0, 40, 10):
    dtm.train(10)
    print(f"Iteration {i+10}, Log-likelihood per word: {dtm.ll_per_word:.4f}")

print("Training abgeschlossen!")
print("Docs per timepoint:", dtm.num_docs_by_timepoint)

# ---------- 9️⃣ TOPWÖRTER ----------
print("\nTop-Wörter der Topics (Timepoint 10):")
for k in range(dtm.k):
    print(f"\nTopic {k}:")
    for word, prob in dtm.get_topic_words(k, timepoint=15, top_n=10):
        print(f"  {word} ({prob:.3f})")
