import pandas as pd
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent.parent
IN_CSV = HERE / "data" / "clean_data" / "train_clean.csv"
OUT_DIR = HERE / "data" / "split_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(IN_CSV, dtype=str).fillna("")

    # Use `transliteration_with_annotations` as input and `translation` as target
    df = df[["transliteration_with_annotations", "translation"]].rename(
        columns={"transliteration_with_annotations": "input_text", "translation": "target_text"}
    )

    # Drop rows missing either field
    df = df[(df["input_text"].str.strip() != "") & (df["target_text"].str.strip() != "")]

    # Reproducible shuffle + 90/10 split without scikit-learn
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = max(1, int(len(df_shuffled) * 0.1))
    val = df_shuffled.iloc[:n_val]
    train = df_shuffled.iloc[n_val:]

    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "validation.jsonl"

    train.to_json(train_path, orient="records", lines=True, force_ascii=False)
    val.to_json(val_path, orient="records", lines=True, force_ascii=False)

    meta = {"train_rows": len(train), "validation_rows": len(val)}
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote {train_path} ({len(train)}) and {val_path} ({len(val)})")

if __name__ == '__main__':
    main()
