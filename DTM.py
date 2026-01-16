# --- Imports ---
import pandas as pd
import re
import nltk
import tomotopy as tp
from nltk.corpus import stopwords
from tqdm import tqdm
import random
import numpy as np 

# --- 0️⃣ NLTK Stopwords herunterladen ---
nltk.download("stopwords")

# --- 1️⃣ CSV laden ---
df = pd.read_csv(
    "filtered_labels_output.csv",
    sep=";",  # Semikolon als Trenner
    quotechar='"',
    on_bad_lines="skip",
    engine="python",
)
df = df[["text", "label", "class"]].dropna()

# --- 2️⃣ OCR-Normalisierung ---
def normalize_ocr(text):
    text = str(text)
    text = text.replace("ſ", "s")  # langes s

    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ß": "ss",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # nur Buchstaben behalten
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(normalize_ocr)

# --- 3️⃣ Jahr aus Label extrahieren ---
def extract_year(label):
    m = re.search(r"(18|19|20)\d{2}", str(label))
    if m:
        return int(m.group())
    return None

df["year"] = df["label"].apply(extract_year)
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# --- 4️⃣ Tokenisierung + Stopwords ---
stop_words = set(stopwords.words("german"))
# Eigene Stopwords ergänzen

custom_stopwords = {"dass","daß","der","die","das","und","mit","von","zu","den","des",
    "im","für", "fuer", "auf","ist","nicht",
    "aber", "als", "auch", "aus", "bei", "bis",
    "worden", "wurde", "wurden",
    "fuer", "ueber", "wegen",
    "waere", "sei", "wuerde",
    "mehr", "immer", "erst", "schon",
    "grosse", "grossen", "kleinen", "ganz",
    "teil", "zeit", "jahre", "ueber", "waehrend", 
    "koennen", "kommen", "muessen",
    "the", 
    "ver", "ven",  
}


stop_words.update(custom_stopwords)

def tokenize(text):
    return [w for w in text.split() if w not in stop_words and len(w) > 2]

df["tokens"] = df["clean_text"].apply(tokenize)

# --- 5️⃣ Kurze Dokumente rausfiltern ---
min_tokens = 3
df_filtered = df[df["tokens"].apply(len) >= min_tokens].copy()

# --- 5.1️⃣ Häufigkeiten berechnen ---
from collections import Counter

all_tokens = [token for doc in df_filtered["tokens"] for token in doc]
freq = Counter(all_tokens)

print("Vokabulargröße (vor Hapax-Filter):", len(freq))

# --- 5.2️⃣ Hapax-Legomena entfernen ---
hapax = {w for w, c in freq.items() if c == 1}

df_filtered["tokens"] = df_filtered["tokens"].apply(
    lambda doc: [w for w in doc if w not in hapax]
)

# --- 5.3️⃣ Dokumente erneut prüfen ---
df_filtered = df_filtered[df_filtered["tokens"].apply(len) >= min_tokens].copy()

# Optional: Testlauf mit begrenzter Anzahl Dokumente
df_sample = df_filtered.head(150_000)

print("Ursprüngliche Dokumente:", len(df))
print("Nach Filterung:", len(df_filtered))
print("Testlauf:", len(df_sample))

# --- 6️⃣ Dynamic Topic Model erstellen ---
years = sorted(df_filtered["year"].unique())
year_to_tp = {year: idx for idx, year in enumerate(years)}


all_tokens = [token for doc in df_filtered["tokens"] for token in doc]


vocab = set(all_tokens)

print("Vokabulargröße:", len(vocab))

print("Top 50 häufigste Wörter:")
print(freq.most_common(25))

print("\nAnzahl Wörter, die nur 1x vorkommen:", sum(1 for w,c in freq.items() if c == 1))




random.seed(42)
np.random.seed(42)


dtm = tp.DTModel(
    k=15,                # Anzahl Topics
    t=len(years),        # Zeitpunkte
    min_cf=10,           # minimale Termfrequenz
    min_df=3,            # Term muss in mindestens 3 Dokumenten vorkommen
    tw=tp.TermWeight.IDF # IDF-Gewichtung
)

# --- 7️⃣ Dokumente hinzufügen ---
for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    dtm.add_doc(row["tokens"], timepoint=year_to_tp[row["year"]])

dtm.burn_in = 100  # optional

# --- 8️⃣ DTM trainieren ---
print("Starte Training...")
for i in range(0, 50, 10):  # kleine Schritte für Kontrolle
    dtm.train(10)
    print(f"Iteration {i+10}, Log-likelihood per word: {dtm.ll_per_word:.4f}")

print("Training abgeschlossen!")
print("Docs per timepoint:", dtm.num_docs_by_timepoint)

# --- 9️⃣ Top-Wörter pro Topic für ersten Timepoint ---
print("\nTop-Wörter der Topics (Timepoint 10):")
for k in range(dtm.k):
    print(f"\nTopic {k}:")
    for word, prob in dtm.get_topic_words(k, timepoint=10, top_n=10):
        print(f"  {word} ({prob:.3f})")
