import spacy
from tqdm import tqdm
from collections import Counter
import pandas as pd
import numpy as np
import random
import sys

def spacy_tokenize_lemmatize(
    docs,
    nlp,
    batch_size=1000,
    n_process=1,
    token_stopwords=set(),
    lemma_stopwords=set(),
    min_token_len=3,
    verbose=True
):

    # Token-Stopwords in nlp.vocab markieren
    for w in token_stopwords:
        nlp.vocab[w].is_stop = True

    # Tokenisierung + Lemmatization
    all_tokens = []
    iterator = nlp.pipe(docs, batch_size=batch_size, n_process=n_process)
    if verbose:
        iterator = tqdm(iterator, total=len(docs))

    for doc in iterator:
        tokens = []
        append = tokens.append

        for token in doc:
            if not token.is_alpha or token.is_stop:
                continue

            lemma = token.lemma_.lower()
            if len(lemma) >= min_token_len and lemma not in lemma_stopwords:
                append(lemma)

        all_tokens.append(tokens)

    return all_tokens

if __name__ == "__main__":
    # Sample DataFrame laden
    df = pd.read_parquet("input/normalised_input.parquet")

    # spaCy Pipeline
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])

    # Stopwords
    token_stopwords = {
        "dass","daß","der","die","das","und","mit","von","zu","den","des",
        "im","für","fuer","auf","ist","nicht","aber","als","auch","aus","bei","bis",
        "worden","wurde","wurden","fuer","ueber","wegen",
        "waere","sei","wuerde","mehr","immer","erst","schon",
        "grosse","grossen","kleinen","ganz","teil","zeit","jahre","ueber","waehrend", 
        "koennen","kommen","muessen",
        "the","ver","ven", "sch"
    }

    lemma_stopwords = {
        # Hilfsverben / Modalverben
        "sein", "haben", "werden", "kommen", "koennen", "muessen",
        "sollen", "wollen", "duerfen", "moegen",

        # Allgemeine Verben / häufige Funktionswörter
        "lassen", "halten", "geben", "finden", "bringen", "bleiben", "treten",
        "setzen", "treten", "bilden", "liegen", "folgen", "stehen", "bitten",

        # Pronomen
        "ich", "du", "er", "sie", "es", "wir", "ihr", "man", "mein", "dein", "sein", "ihr", "uns", "euch",

        # Artikel
        "der", "die", "das", "ein", "eine", "einer", "dem", "den", "des",

        # Präpositionen
        "in", "auf", "an", "zu", "von", "mit", "fuer", "ueber", "unter", "nach", "vor", "bei", "aus", "gegen",

        # Konjunktionen / Füllwörter
        "und", "oder", "aber", "wenn", "weil", "da", "dass", "als", "wie", "so", "also", "nur", "noch", "schon", "mal", "daher",

        # Zeit / Zahlen / Mengen
        "zeit", "jahr", "monat", "tag", "stunde", "viel", "wenig", "erste", "letzte", "weit", "alle",

        # Adverbien / andere häufige Wörter
        "fast", "direkt", "lich", "mehr", "sehr", "einfach", "wirklich", "oben", "unten", "hier", "dort",

        # Sonstige
        "andr", "fur", "abt", "tie", "art", "usw", "fast", "sodass", "unsr"
    }



    # Tokenisierung + Lemmatization
    df["tokens"] = spacy_tokenize_lemmatize(
        df["text"].tolist(),
        nlp,
        batch_size=5000,
        n_process=3,
        token_stopwords=token_stopwords,
        lemma_stopwords=lemma_stopwords,
        min_token_len=3,
    )

    df.to_parquet("input/tokenized_corpus.parquet")
