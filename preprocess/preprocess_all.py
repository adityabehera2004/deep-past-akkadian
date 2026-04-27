import os
import importlib.util
from pathlib import Path
import pandas as pd
import re

# load preprocess module
module_path = os.path.join(os.path.dirname(__file__), "preprocess.py")
spec = importlib.util.spec_from_file_location("preprocess", module_path)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

# Local lookup + annotator re-used from preprocess_test.py style
def lookup_definition(tok):
    d = preprocess.get_akkadian_definition(tok)
    if not d:
        return None
    d = d.strip()
    if len(d) >= 2 and d[0] == '"' and d[-1] == '"':
        d = d[1:-1]
    # skip overly long multi-definition entries
    if d.count(';') > 3:
        return None
    return d

def annotate_sentence(sentence):
    if pd.isna(sentence) or not isinstance(sentence, str):
        return ''
    parts = []
    for tok in sentence.split():
        stripped = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', tok)
        definition = lookup_definition(stripped)
        if definition:
            parts.append(f"{tok} {{{definition}}}")
        else:
            parts.append(tok)
    return ' '.join(parts)

CLEAN_ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
RAW = CLEAN_ROOT / "data" / "raw_data"
CLEAN_DIR = CLEAN_ROOT / "data" / "clean_data"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IN = RAW / "train.csv"
TEST_IN = RAW / "test.csv"
SENT_IN = RAW / "Sentences_Oare_FirstWord_LinNum.csv"

TRAIN_OUT = CLEAN_DIR / "train_clean.csv"
TEST_OUT = CLEAN_DIR / "test_clean.csv"


def make_train_clean():
    print(f"Reading {TRAIN_IN}")
    train_df = pd.read_csv(TRAIN_IN, dtype=str)
    print(f"Reading {SENT_IN}")
    sentences_df = pd.read_csv(SENT_IN, dtype=str)

    # align
    sent_level = preprocess.align_to_sentence_level(train_df, sentences_df)

    rows = []
    # group original train by oare_id for fallback
    train_map = {r.oare_id: r for r in train_df.itertuples()}

    for idx, r in sent_level.iterrows():
        trans = r.get('transliteration', '')
        trad = r.get('translation', '')
        trans_c = preprocess.preprocess_akkadian_text(trans)
        trans_annot = annotate_sentence(trans_c)
        trad_c = preprocess.preprocess_english_text(trad)
        rows.append({
            'oare_id': r.get('oare_id'),
            'sentence_uuid': r.get('sentence_id'),
            'transliteration': trans_c,
            'transliteration_with_annotations': trans_annot,
            'translation': trad_c
        })

    # find train docs missing in sent_level and add as single sentence
    sent_oare = set(sent_level['oare_id'].unique())
    for r in train_df.itertuples():
        if r.oare_id not in sent_oare:
            trans = r.transliteration if hasattr(r, 'transliteration') else ''
            trad = r.translation if hasattr(r, 'translation') else ''
            rows.append({
                'oare_id': r.oare_id,
                'sentence_uuid': None,
                'transliteration': preprocess.preprocess_akkadian_text(trans),
                'translation': preprocess.preprocess_english_text(trad)
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(TRAIN_OUT, index=False)
    print(f"Saved train clean -> {TRAIN_OUT}")


def make_test_clean():
    print(f"Reading {TEST_IN}")
    test_df = pd.read_csv(TEST_IN, dtype=str)
    # Clean transliteration field
    if 'transliteration' in test_df.columns:
        test_df['transliteration_clean'] = test_df['transliteration'].apply(preprocess.preprocess_akkadian_text)
    else:
        test_df['transliteration_clean'] = ''
    # add annotated column
    test_df['transliteration_with_annotations'] = test_df['transliteration_clean'].apply(annotate_sentence)
    test_df.to_csv(TEST_OUT, index=False)
    print(f"Saved test clean -> {TEST_OUT}")


if __name__ == '__main__':
    make_train_clean()
    make_test_clean()
