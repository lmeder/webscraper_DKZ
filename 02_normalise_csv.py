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

# ---------- SONDERZEICHENDICHTE ----------
def special_char_density(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    total_chars = len(text)
    special_chars = len(re.findall(r'[^A-Za-z0-9äöüÄÖÜßſ\s]', text))  # alles außer Buchstaben, Zahlen, Leerzeichen
    return special_chars / total_chars
    
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

    # ---------- Jahr 1883 entfernen ----------
    df = df[df["year"] != 1883]
    print(f"Nach Entfernen von 1883: {len(df)} Zeilen")

    df["region_length"] = df.text.str.len()
    df['special_char_density'] = df['text'].apply(special_char_density)

    # Filtern kurzer Regions um Werbung auszuschließen
    df = df[df.region_length > 500]
    df = df[df.special_char_density < 0.1]
    # Test ob es sinnvoll ist Preis rauszufiltern

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
