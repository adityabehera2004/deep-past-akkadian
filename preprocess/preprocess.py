import pandas as pd
import re
from fractions import Fraction
import csv
from pathlib import Path
import urllib.parse
import urllib.request
import json
import ssl

# These are just all the helper functions used by the preprocess_*.py scripts

_MEANING_CACHE = {}

def process_fractions_dynamic(text_to_process):
    if pd.isna(text_to_process) or not isinstance(text_to_process, str):
        return str(text_to_process) if not pd.isna(text_to_process) else ""
    # Step0: Preprocess
    # --- 0.1 Clean number boundaries ---
    text_to_process = re.sub(r'\((\+?\d+)\)', r'\1', text_to_process)
    text_to_process = re.sub(r'\[(\+?\d+)\]', r'\1', text_to_process)
    text_to_process = re.sub(r'˹(\+?\d+)˺', r'\1', text_to_process)
    text_to_process = re.sub(r'⌈(\+?\d+)⌋', r'\1', text_to_process)
    text_to_process = re.sub(r'「(\+?\d+)」', r'\1', text_to_process)
    text_to_process = re.sub(r'⸢(\+?\d+)⸣', r'\1', text_to_process)

    # --- 0.2 Convert X+X format additions ---
    def calculate_addition(match):
        try:
            val1 = float(match.group(1))
            val2 = float(match.group(2))
            result = val1 + val2
            return "{:.5f}".format(result).rstrip('0').rstrip('.') if result % 1 != 0 else str(int(result))
        except ValueError:
            return match.group(0)
    addition_pattern = r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)'
    while True:
        new_text = re.sub(addition_pattern, calculate_addition, text_to_process)
        if new_text == text_to_process:
            break
        text_to_process = new_text

    # --- 0.3 Replace Roman numeral months ---
    roman_to_arabic = {
        "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6",
        "VII": "7", "VIII": "8", "IX": "9", "X": "10", "XI": "11", "XII": "12"
    }
    roman_pattern = r'\b(Month|month)\s+(VIII|XII|VII|III|XI|IX|VI|IV|II|X|V|I)\b'
    def replace_roman(match):
        return f"{match.group(1)} {roman_to_arabic[match.group(2)]}"
    text_to_process = re.sub(roman_pattern, replace_roman, text_to_process)

    # Step1: Convert X / X fractions
    fraction_map = {
        r"1\s*/\s*6": "⅙",
        r"1\s*/\s*4": "¼",
        r"1\s*/\s*3": "⅓",
        r"1\s*/\s*2": "½",
        r"2\s*/\s*3": "⅔",
        r"3\s*/\s*4": "¾",
        r"5\s*/\s*6": "⅚",
    }
    for pattern, char in fraction_map.items():
        text_to_process = re.sub(pattern, char, text_to_process)

    def calc_fraction(match):
        groups = match.groups()
        if groups[0]:
            integer_part = int(groups[0])
            num = int(groups[1])
            den = int(groups[2])
            val = integer_part + (num / den)
        else:
            num = int(groups[3])
            den = int(groups[4])
            val = num / den
        return "{:.5f}".format(val).rstrip('0').rstrip('.') if val % 1 != 0 else str(int(val))
    
    fraction_pattern = r'(\d+)\s+(\d+)/(\d+)|(\d+)/(\d+)'   
    text_to_process = re.sub(fraction_pattern, calc_fraction, text_to_process)
    
    # Step2: Convert decimal fractions to Unicode
    text_to_process = re.sub(r'(\d+\.\d{5})\d+', r'\1', text_to_process)

    fraction_lookup = {
            (1, 6): "⅙", (1, 4): "¼", (1, 3): "⅓", (1, 2): "½",
            (2, 3): "⅔", (3, 4): "¾", (5, 6): "⅚",
    }
    def replacer(match):
        full_str = match.group(0)
        try:
            val = float(full_str)
            integer_part = int(val)
            decimal_part = val - integer_part
            if decimal_part < 0.0001: 
                return str(integer_part)
            frac = Fraction(decimal_part).limit_denominator(12) 
            unicode_frac = fraction_lookup.get((frac.numerator, frac.denominator))
            if unicode_frac:
                if integer_part == 0:
                    return unicode_frac
                else:
                    return f"{integer_part} {unicode_frac}"
            return full_str 
        except ValueError:
            return full_str
            
    processed_text = re.sub(r'\b\d+\.\d+\b', replacer, text_to_process)
    
    # Step3: Robustness check
    processed_text = re.sub(r'(\d)([¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])', r'\1 \2', processed_text)
    
    return processed_text

def preprocess_akkadian_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Artifact removal
    pattern = r"\s*Seal Impression [A-Z]\s*"
    text = re.sub(pattern, "", text)
    text = re.sub(r"Seal Impression", "", text)
    text = re.sub(r"broken", "xxx", text)
    text = re.sub(r"------------", "", text)
    text = re.sub(r"---- Single Ruling ----", "", text)
    text = re.sub(r"---- Double Ruling ----", "", text)
    
    # Character map
    char_map = str.maketrans({
        "X": "x", "×": "x", "ā": "a", "Ş": "Ṣ", "ş": "ṣ",
        "Ș": "Ṣ", "ș": "ṣ", "Ț": "Ṭ", "ț": "ṭ", "İ": "I",
        "Ī": "I", "ī": "i", "Î": "I", "î": "i", "ı": "i",
        "h": "ḫ", "H": "Ḫ", "ḥ": "ḫ", "Ḥ": "Ḫ", "=": "-",
        "(": "{", ")": "}", "—": "-", "–": "-",
    })
    text = text.translate(char_map)
    text = text.replace("ᵈ", r"{d}").replace("ᵏⁱ", r"{ki}")
    text = text.replace(r"{{", "{").replace(r"}}", "}")
    text = text.replace("??", "?").replace("!!", "!").replace("//", "/")
    text = text.replace("-/", "-")
    text = text.replace("{?}", "").replace("{!}", "")
    text = text.replace("(?)", "").replace("(!)", "")
    
    # Upper/bad char removal
    upper_chars = ["⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹","⁻","°", "˚"]
    for ch in upper_chars: text = text.replace(ch, "")
    bad_chars = """"ʺʾ’´ʼˊ˘ˋˇˈʿʹ'`^˹˺⸢⸣⌜⌝「」⌈⌋⌊⌉˻˼⸤⸥≪≫《》⟪⟫«»〈〉⟨⟩˃‹›⁽⁾ˁˀ"""
    for ch in bad_chars: text = text.replace(ch, "")
    bad_chars = """!?⸮ʔ⸮,/;˷˴|:⁺*"""
    for ch in bad_chars: text = text.replace(ch, " ")

    # Fractions
    text = process_fractions_dynamic(text)
    
    # Subscript handling for standard Akkadian notations
    letters_str = "a-zA-ZšŠṣṢṭṬḫḪ"
    subscript_trans = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    def replace_group_with_subscript(match):
        return match.group(0).translate(subscript_trans)
    text = re.sub(f"(?<=[{letters_str}])\d+", replace_group_with_subscript, text)

    # Dropping annotations
    text = re.sub(r'\(\s*(?:fem\.|sing\.|pl\.|plural|f\.|plur\.)(?:\s+(?:fem\.|sing\.|pl\.|plural|f\.|plur\.))*\s*\)', '', text)
    text = re.sub(r'\(\s*(?:K|Rs|lR)(?:\s+(?:K|Rs|lR))*\s*\)', '', text)
    text = re.sub(r'\b(?:le\.e\.|lo\.e\.|l\.o\.e\.|obv\.|rev\.|r\.e\.|u\.e\.)\s*', '', text)

    # Gap processing
    text = re.sub(r'\bPN\b|\bPn\b', '<gap>', text)
    text = re.sub(r'\bbreak\b|\bbroken\b', '', text)
    text = re.sub(r'\[[\sx?]*\]', '[xxx]', text)
    for ch in "[]": text = text.replace(ch, "")
    text = re.sub(r'(?:\.\s+){2,}\.', lambda m: m.group(0).replace(' ', ''), text)
    text = re.sub(r'\.{3,}(\s+\.{3,})*', '<gap>', text)
    text = re.sub(r'……|…', '<gap>', text)
    text = re.sub(r'xx+', '<gap>', text)
    text = re.sub(r'\bx\b', '<gap>', text)
    text = text.replace("{large break}", "<gap>")
    text = re.sub(r'\{\d+\s+broken\s+lines\}', '<gap>', text)
    
    text = text.replace("<gap>", "\x00GAP\x00")
    for ch in "<>": text = text.replace(ch, "")
    text = text.replace("\x00GAP\x00", " <gap> ")
    
    text = re.sub(r"\s+", " ", text).strip().strip("-")
    text = text.replace(" -", "-").replace("- ", "-")
    text = text.replace(" .", ".").replace(". ", ".")
    text = text.replace("> .", ">.").replace("> :", ">:").replace("> ’", ">’")

    text = re.sub(r'<gap>([ \-\t]*<gap>)+', '<gap>', text)
    text = text.replace("{ <gap> }", "<gap>").replace("[ <gap> ]", "<gap>")
    
    return text

def preprocess_english_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    
    char_map = str.maketrans({
        "X": "x", "ʾ": "'", "Ş": "Ṣ", "ş": "ṣ",
        "ș": "ṣ", "Ț": "Ṭ", "ț": "ṭ", "ḫ": "h",
        "Ḫ": "H", "ḥ": "ḫ", "Ḥ": "Ḫ",
    })
    text = text.translate(char_map)
    text = text.replace("{?}", "").replace("{!}", "")
    text = text.replace("(?)", "").replace("(!)", "")
    
    upper_chars = ["⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹","⁻","°","˚"]
    for ch in upper_chars: text = text.replace(ch, "")
    lower_chars = ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"]
    for ch in lower_chars: text = text.replace(ch, "")
    bad_chars = """"ʺˊ˘ˋˇˈʿ`^˹˺⸢⸣⌜⌝「」⌈⌋⌊⌉˻˼⸤⸥≪≫《》⟪⟫«»〈〉⟨⟩˃‹›⁽⁾ˁˀ"""
    for ch in bad_chars: text = text.replace(ch, "")
    for ch in """ʹ'ʾ´ʼ""": text = text.replace(ch, "’")

    text = process_fractions_dynamic(text)

    text = re.sub(r'\(\s*(?:fem\.|sing\.|pl\.|plural|f\.|plur\.)(?:\s+(?:fem\.|sing\.|pl\.|plural|f\.|plur\.))*\s*\)', '', text)
    text = re.sub(r'\(\s*(?:K|Rs|lR)(?:\s+(?:K|Rs|lR))*\s*\)', '', text)
    text = re.sub(r'\b(?:le\.e\.|lo\.e\.|l\.o\.e\.|obv\.|rev\.|r\.e\.|u\.e\.)\s*', '', text)

    text = re.sub(r'\bPN\b|\bPn\b', '<gap>', text)
    text = re.sub(r'\[[\sx?]*\]', '[xxx]', text)
    text = re.sub(r'(?:\.\s+){2,}\.', lambda m: m.group(0).replace(' ', ''), text)
    text = re.sub(r'\.{3,}(\s+\.{3,})*', '<gap>', text)
    text = re.sub(r'……|…', '<gap>', text)
    text = re.sub(r'xx+', '<gap>', text, flags=re.I)
    text = re.sub(r'\bx\b', '<gap>', text, flags=re.I)
    text = text.replace("{large break}", "<gap>")
    text = re.sub(r'\{\d+\s+broken\s+lines\}', '<gap>', text)
    
    text = text.replace("<gap>", "\x00GAP\x00")
    for ch in '˹˺⸢⸣「」⌈⌋⌊|[]': text = text.replace(ch, "")
    for ch in "/*+": text = text.replace(ch, " ")
    text = text.replace("\x00GAP\x00", " <gap> ")

    text = re.sub(r"\s+", " ", text).strip().strip("-")
    text = text.replace(" -", "-").replace("- ", "-")
    text = text.replace("> .", ">.").replace("> :", ">:")

    text = re.sub(r'<gap>([ \-\t]*<gap>)+', '<gap>', text)
    text = text.replace("{ <gap> }", "<gap>").replace("[ <gap> ]", "<gap>")
    
    return text

def align_to_sentence_level(train_df, sentences_df):
    """
    Aligns document-level transliterations in train_df to the sentence-level translations
    provided in sentences_df (Sentences_Oare_FirstWord_LinNum.csv).
    
    Returns a dataframe where each row is a sentence containing:
      - oare_id / text_uuid
      - sentence_uuid
      - transliteration (sentence-level)
      - translation (sentence-level)
    """
    aligned_data = []

    # Iterate grouped by text document
    for text_uuid, group in sentences_df.groupby("text_uuid"):
        # Sort by sentence order using numeric first_word_number, not string sort
        group = group.copy()
        group["first_word_number"] = pd.to_numeric(group["first_word_number"], errors="coerce")
        group = group.sort_values(by="first_word_number", ascending=True)
        
        # Get full document text
        doc_row = train_df[train_df["oare_id"] == text_uuid]
        if doc_row.empty:
            continue
            
        full_translit = str(doc_row.iloc[0]["transliteration"]).split()
        
        # To align transliterations with sentences, we use the `first_word_number`
        group = group.reset_index(drop=True)
        
        for i, row in group.iterrows():
            start_idx = int(row["first_word_number"]) - 1 if pd.notna(row["first_word_number"]) else 0
            
            # End index is the start of the next sentence, or end of document
            if i + 1 < len(group) and pd.notna(group.iloc[i+1]["first_word_number"]):
                end_idx = int(group.iloc[i+1]["first_word_number"]) - 1
            else:
                end_idx = len(full_translit)
                
            sentence_translit = " ".join(full_translit[start_idx:end_idx])
            sentence_translation = row["translation"] if pd.notna(row["translation"]) else ""
            
            aligned_data.append({
                "oare_id": text_uuid,
                "sentence_id": row["sentence_uuid"],
                "transliteration": sentence_translit,
                "translation": sentence_translation
            })

    return pd.DataFrame(aligned_data)

def get_akkadian_definition(token):
    """Return the definition string for an Akkadian token from the local CSV.

    This function reads from data/clean_data/definitions.csv only.
    It does NOT call the remote API.

    Returns None if not found.
    """
    if not token:
        return None
    token = str(token).strip()
    
    # Load definitions.csv into memory (lazy load, cached globally)
    global _MEANING_CACHE
    if not _MEANING_CACHE:
        csv_path = Path(__file__).parent.parent / "data" / "clean_data" / "definitions.csv"
        if csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row:
                            continue
                        w = (row.get('word') or '').strip()
                        d = (row.get('definition') or '').strip()
                        if not w:
                            continue
                        _MEANING_CACHE[w] = d
                        _MEANING_CACHE[w.lower()] = d
                        stripped = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', w)
                        if stripped:
                            _MEANING_CACHE[stripped] = d
                            _MEANING_CACHE[stripped.lower()] = d
            except Exception:
                pass
    
    # Look up in the cached mapping
    if token in _MEANING_CACHE:
        return _MEANING_CACHE[token]
    if token.lower() in _MEANING_CACHE:
        return _MEANING_CACHE[token.lower()]
    stripped = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', token)
    if stripped in _MEANING_CACHE:
        return _MEANING_CACHE[stripped]
    if stripped.lower() in _MEANING_CACHE:
        return _MEANING_CACHE[stripped.lower()]
    return None

if __name__ == "__main__":
    # Example usage:
    # train_df = pd.read_csv("data/raw_data/train.csv")
    # sentences_df = pd.read_csv("data/raw_data/Sentences_Oare_FirstWord_LinNum.csv")
    # sentence_level_df = align_to_sentence_level(train_df, sentences_df)
    # sentence_level_df["transliteration"] = sentence_level_df["transliteration"].apply(preprocess_akkadian_text)
    # sentence_level_df["translation"] = sentence_level_df["translation"].apply(preprocess_english_text)
    # sentence_level_df.to_csv("data/train_sentence_level_clean.csv", index=False)
    pass
