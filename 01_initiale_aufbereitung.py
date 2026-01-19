import json
import pandas as pd
import glob
import os

# Ordner mit JSON-Dateien
json_folder = "output/"

rows_label = []
rows_page = []

# Alle JSON-Dateien im Ordner sammeln
for filepath in glob.glob(os.path.join(json_folder, "*.json")):
    with open(filepath, "r") as f:
        data = json.load(f)

    structures = data.get("structures", [])

    for struct in structures:
        try:
            nummern = [int(nummer.split("/")[-1]) for nummer in struct["canvases"]]
            label = struct["label"]

            rows_label.append(
                {
                    "source_file": os.path.basename(filepath),
                    "label": label,
                    "nummern": nummern,
                }
            )
        except Exception:
            pass

    sequences = data.get("sequences", [])

    for canvas in sequences[0]["canvases"]:
        try:
            id = canvas["@id"].split("/")[-1]
            page = canvas["label"]
            
            rows_page.append(
                    {
                        "source_file": os.path.basename(filepath),
                        "nummern": int(id),
                        "page": page,
                    }
            )
        except Exception:
            pass


# DataFrame bauen
df_labels = (
    pd.DataFrame(rows_label)
    .explode("nummern")
    .drop_duplicates(subset=["source_file", "nummern"], keep="last")
    .reset_index(drop=True)
)

df_page = pd.DataFrame(rows_page).drop_duplicates("nummern", keep="last")


df_csv = pd.read_csv("input/text_export_final.csv", sep=";", encoding="utf-8")

df_csv_merged = df_csv.merge(df_labels, left_on="path", right_on="nummern", how="left")
df_csv_merged = df_csv_merged.merge(df_page[["nummern", "page"]], on="nummern", how="left")

df_csv_merged = df_csv_merged.drop("nummern", axis=1)
df_csv_merged.to_parquet("input/input_complete_dkz.parquet")
