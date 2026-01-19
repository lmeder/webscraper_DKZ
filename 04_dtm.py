# --- Imports ---
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import tomotopy as tp
from collections import Counter
import datetime
import os

def remove_hapax(docs):
    freq = Counter(token for doc in docs for token in doc)
    hapax = {token for token, count in freq.items() if count == 1}
    return [[token for token in doc if token not in hapax] for doc in docs]


def remove_stopwords(docs, stopwords):
    stopwords = set(stopwords)  # in Set umwandeln für schnellere Suche
    return [[token for token in doc if token not in stopwords] for doc in docs]


df = pd.read_parquet("input/tokenized_corpus.parquet")

# ---  Kurze Dokumente rausfiltern ---
min_tokens = 3
df_filtered = df[df["tokens"].apply(len) >= min_tokens].copy()

df_filtered["tokens"] = remove_hapax(df_filtered["tokens"])

stop_words = ["scheinen", "suchen", "nehmen", "erscheinen", "sch", "sehen", "deutsch", "herr", "fres"]
df_filtered["tokens"] = remove_stopwords(df_filtered["tokens"], stop_words)


# --- 5. Dokumente erneut prüfen ---
df_filtered = df_filtered[df_filtered["tokens"].apply(len) >= min_tokens].copy()



print("Ursprüngliche Dokumente:", len(df))
print("Nach Filterung:", len(df_filtered))

# ---------- DTM SETUP ----------
years = sorted(df_filtered["year"].unique())
year_to_tp = {year: idx for idx, year in enumerate(years)}


# --- Parameter merken ---
k = 10
num_timepoints = len(years)
min_cf = 30
min_df = 5
term_weight = "IDF"
workers = 1
seed = 42
total_iters = 600
step = 50

dtm = tp.DTModel(
    k=k,
    t=num_timepoints,
    min_cf=min_cf,
    min_df=min_df,
    tw=tp.TermWeight.IDF,
    seed=seed
)

# ---------- DOKUMENTE HINZUFÜGEN ----------
for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    dtm.add_doc(row["tokens"], timepoint=year_to_tp[row["year"]])

dtm.burn_in = 100

# ---------- TRAINING ----------
print("Starte Training...")
for i in range(0, total_iters, step):
    dtm.train(step, workers=workers)
    print(f"Iteration {i+step:4d} | Log-likelihood per word: {dtm.ll_per_word:.4f}")


print("Training abgeschlossen!")
print("Docs per timepoint:", dtm.num_docs_by_timepoint)


# Topic-Anteile über alle Dokumente
topic_shares_global = np.zeros(dtm.k)
for doc in dtm.docs:
    topic_shares_global += doc.get_topic_dist()
topic_shares_global /= topic_shares_global.sum()

# Topic-Anteile pro Zeitpunkt (Jahr)
num_docs_by_tp = dtm.num_docs_by_timepoint
topic_shares_by_time = np.zeros((num_timepoints, dtm.k))
doc_idx = 0
for t, n_docs in enumerate(num_docs_by_tp):
    for _ in range(n_docs):
        topic_shares_by_time[t] += dtm.docs[doc_idx].get_topic_dist()
        doc_idx += 1
    topic_shares_by_time[t] /= topic_shares_by_time[t].sum()


# Speichert das trainierte Modell
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

dtm.save(os.path.join("dtm_models", f"dtm_model_{timestamp}.bin"))
print("Modell gespeichert als dtm_model.bin")

output_file = os.path.join("dtm_runs", f"dtm_summary_{timestamp}.txt")

with open(output_file, "w", encoding="utf-8") as f:
    # --- Modellparameter ---
    f.write("===== DTM Modellparameter =====\n")
    f.write(f"Anzahl Themen (k): {k}\n")
    f.write(f"Anzahl Zeitpunkte (t): {num_timepoints}\n")
    f.write(f"min_cf: {min_cf}\n")
    f.write(f"min_df: {min_df}\n")
    f.write(f"workers: {workers}\n")
    f.write(f"workers: {total_iters}\n")
    f.write(f"seed: {seed}\n")
    f.write(f"TermWeight: {term_weight}\n")
    f.write(f"Burn-in: {dtm.burn_in}\n")
    f.write(f"Log-likelihood per word: {dtm.ll_per_word:.4f}\n")
    f.write(f"Dokumente pro Zeitpunkt: {dtm.num_docs_by_timepoint}\n\n")

    f.write("===== Topic-Anteile (global) =====\n")
    for k_idx, share in enumerate(topic_shares_global):
        f.write(f"Topic {k_idx}: {share:.4f}\n")
    f.write("\n")

    f.write("===== Topic-Anteile pro Zeitpunkt =====\n\n")

    # Header
    header = "Timepoint".ljust(12)
    for k_idx in range(dtm.k):
        header += f"T{k_idx}".rjust(10)
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")

    # Rows
    for t in range(num_timepoints):
        row = f"{t}".ljust(12)
        for k_idx in range(dtm.k):
            row += f"{topic_shares_by_time[t, k_idx]:.4f}".rjust(10)
        f.write(row + "\n")

    f.write("\n")


    # --- Top-Wörter pro Topic und Zeitpunkt ---
    f.write("===== Top-Wörter der Topics =====\n\n")
    for k_idx in range(dtm.k):
        f.write(f"--- Topic {k_idx} ---\n")
        for t in range(num_timepoints):
            top_words = dtm.get_topic_words(k_idx, timepoint=t, top_n=10)
            top_words_str = ", ".join([f"{word}({prob:.3f})" for word, prob in top_words])
            f.write(f"Timepoint {t}: {top_words_str}\n")
        f.write("\n")

print(f"Modellübersicht und Top-Wörter wurden in '{output_file}' gespeichert.")

