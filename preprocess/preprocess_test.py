import os
import importlib.util
import pandas as pd
import re

# Load preprocess module by path
module_path = os.path.join(os.path.dirname(__file__), "preprocess.py")
spec = importlib.util.spec_from_file_location("preprocess", module_path)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw_data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SENT_CSV = os.path.join(DATA_DIR, "Sentences_Oare_FirstWord_LinNum.csv")


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    sentences_df = pd.read_csv(SENT_CSV)
    #docs = train_df.head(5)
    docs = train_df.sample(5)
    # build sentence-level alignment once
    sentence_df = preprocess.align_to_sentence_level(train_df, sentences_df)

    # Centralized definition lookup used by both branches below.
    def lookup_definition(tok):
        d = preprocess.get_akkadian_definition(tok)
        if not d:
            return None
        d = d.strip()
        if len(d) >= 2 and d[0] == '"' and d[-1] == '"':
            d = d[1:-1]
        # If the definition contains more than 3 semicolons, skip annotating it (too many definitions to be useful)
        if d.count(';') > 3:
            return None
        return d

    # Centralized annotator used for cleaned/annotated output
    def annotate_sentence(sentence):
        parts = []
        for tok in sentence.split():
            stripped = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', tok)
            definition = lookup_definition(stripped)
            if definition:
                parts.append(f"{tok} {{{definition}}}")
            else:
                parts.append(tok)
        return ' '.join(parts)

    for i, row in docs.iterrows():
        oare_id = row.get('oare_id')
        print(f"=== DOC (oare_id={oare_id}) ===")
        print("Raw Akkadian:\n")
        print(row.get('transliteration', ''))
        print()
        print("Raw English:\n")
        print(row.get('translation', ''))
        print()

        # extract sentence-level rows for this doc
        doc_sents = sentence_df[sentence_df['oare_id'] == oare_id]
        if doc_sents.empty:
            print("(No sentence-level entries found for this document.)")
            # Treat the whole document as one sentence
            print()
            n = 1
            print(f"SENTENCE {n} ORIGINAL")
            print("[AKKADIAN]")
            print(row.get('transliteration', ''))
            print("[ENGLISH]")
            print(row.get('translation', ''))
            print()

            print(f"SENTENCE {n} CLEANED")
            print("[AKKADIAN]")
            a_clean = preprocess.preprocess_akkadian_text(row.get('transliteration', ''))
            print(a_clean)
            print("[ANNOTATED]")
            print(annotate_sentence(a_clean))
            print("[ENGLISH]")
            e_clean = preprocess.preprocess_english_text(row.get('translation', ''))
            print(e_clean)
            print()
            continue

        for sidx, srow in doc_sents.reset_index(drop=True).iterrows():
            n = sidx + 1
            # Original
            print(f"SENTENCE {n} ORIGINAL")
            print("[AKKADIAN]")
            print(srow.get('transliteration', ''))
            print("[ENGLISH]")
            print(srow.get('translation', ''))
            print()

            # Cleaned
            print(f"SENTENCE {n} CLEANED")
            print("[AKKADIAN]")
            a_clean = preprocess.preprocess_akkadian_text(srow.get('transliteration', ''))
            print(a_clean)
            print("[ANNOTATED]")
            print(annotate_sentence(a_clean))
            print("[ENGLISH]")
            e_clean = preprocess.preprocess_english_text(srow.get('translation', ''))
            print(e_clean)
            print()
    # end per-document printing

if __name__ == '__main__':
    main()
