# --- Imports ---
import pandas as pd
import re
from nltk.corpus import stopwords


# ---------- OCR-NORMALISIERUNG ----------
def normalize_ocr(text):
    text = str(text)
    text = text.replace("ſ", "s")
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
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- JAHR AUS LABEL ----------
def extract_year(label):
    m = re.search(r"(18|19|20)\d{2}", str(label))
    if m:
        return int(m.group())
    return None


def simple_corpus_normalization():

    # ---------- CSV LADEN ----------
    path = "input/input_complete_dkz.parquet"
    print("Lade CSV-Datei...")
    input_df = pd.read_parquet(path)
    print(f"Zeilen geladen: {len(input_df)}")

    # ---------- Relevante Spalten + NaNs entfernen ----------
    df = (
        input_df[["path", "region", "text", "label", "class"]]
        .copy()
        .dropna(subset=["path", "region", "text", "label"])
    )
    print(f"Nach DropNA: {len(df)} Zeilen")

    # ---------- OCR Normalisierung ----------
    df["clean_text"] = df["text"].apply(normalize_ocr)
    print("OCR Normalisierung abgeschlossen")

    # ---------- Jahr extrahieren ----------
    df["year"] = df["label"].apply(extract_year)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    print(f"Jahr extrahiert")

    # ---------- Regionen zu Seiten zusammenfassen ----------
    print("Fasse OCR-Regionen pro Seite zusammen ...")
    df_pages = (
        df.groupby(["year", "path"])["clean_text"]
        .apply(lambda parts: " ".join(parts))
        .reset_index()
        .rename(columns={"clean_text": "text"})
    )
    print(f"Ergebnis: {len(df_pages)} Seiten-Dokumente")

    # ---------- Parquet speichern ----------
    df_pages.to_parquet("input/normalised_input.parquet")
    print("Daten als 'normalised_input.parquet' gespeichert")


if __name__ == "__main__":
    simple_corpus_normalization()
