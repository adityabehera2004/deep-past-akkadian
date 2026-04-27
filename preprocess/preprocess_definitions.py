import os
import csv
import sys
import time
from pathlib import Path
import urllib.parse
import urllib.request
import json
import ssl
import re

# Set `EBL_COOKIE` and `EBL_USER_AGENT` in your .env to override
COOKIE = os.getenv("EBL_COOKIE", "_ga=GA1.1.940817892.1777066332; _ga_H9382J0Y6L=GS2.1.s1777066331$o1$g1$t1777066419$j42$l0$h0")
USER_AGENT = os.getenv("EBL_USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36")

# this will go through OA_Lexicon_eBL.csv and only process words (not PN - proper nouns since those don't really have definitions)
# and extract their definitions from the eBL API (technically not using the official API)
# we are using cached cookies so we can just access the API without having an auth token

CLEAN_ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
LEXICON_CSV = CLEAN_ROOT / "data" / "raw_data" / "OA_Lexicon_eBL.csv"
CLEAN_DIR = CLEAN_ROOT / "data" / "clean_data"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = CLEAN_DIR / "definitions.csv"


_OA_LEXICON_CACHE = None

def _load_oa_lexicon():
    global _OA_LEXICON_CACHE
    if _OA_LEXICON_CACHE is not None:
        return _OA_LEXICON_CACHE

    csv_path = LEXICON_CSV
    mapping = {}
    if not csv_path.exists():
        _OA_LEXICON_CACHE = mapping
        return mapping

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 5:
                continue
            if row[0] == "word":
                token = row[1].strip()
                url = row[4].strip()
                if token and url:
                    mapping[token] = url
                    mapping[token.lower()] = url
                    stripped = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', token)
                    mapping[stripped] = url
                    mapping[stripped.lower()] = url

    _OA_LEXICON_CACHE = mapping
    return _OA_LEXICON_CACHE


def fetch_word_definition_data(token, original_url):
    """
    Query the eBL API for a given token and original raw URL.
    Returns dict: {"word","raw_url","api_url","definition"}
    """
    if not token or not original_url or "word=" not in original_url:
        return None

    token = str(token).strip()
    lemma_part = original_url.split("word=")[1]
    lemma = urllib.parse.unquote(lemma_part)

    # Construct exact encoded URLs
    part1_encoded = "word=" + urllib.parse.quote(lemma)
    raw_url = f"https://www.ebl.lmu.de/dictionary?{part1_encoded}"

    query_encoded = urllib.parse.quote(part1_encoded)
    api_url = f"https://www.ebl.lmu.de/api/words?query={query_encoded}"

    headers = {
        "cookie": COOKIE,
        "user-agent": USER_AGENT
    }

    req = urllib.request.Request(api_url, headers=headers)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    meaning_str = ""
    try:
        with urllib.request.urlopen(req, timeout=5, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
            meanings = []
            for item in data:
                if "meaning" in item and item["meaning"]:
                    m_str = item["meaning"]
                    extracted = re.findall(r'"([^\"]*)"', m_str)
                    if extracted:
                        m = ", ".join(extracted)
                    else:
                        m = m_str.replace('"', '')
                    m = m.replace('\\[', '[').replace('\\]', ']')
                    m = m.replace('\\<', '<')
                    meanings.append(m.strip())

            if meanings:
                meaning_str = "; ".join(meanings)
    except Exception:
        # Ignore HTTP/SSL errors and leave meaning_str empty if failed
        pass

    return {
        "word": token,
        "raw_url": raw_url,
        "api_url": api_url,
        "definition": meaning_str
    }

def extract_definitions(limit=None):
    queue = []
    seen = set()
    
    print(f"Reading dictionary metadata from {LEXICON_CSV}...")
    with open(LEXICON_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 5:
                continue
            
            row_type = row[0].strip()
            token = row[1].strip()
            url = row[4].strip()
            
            # Use 'word' type rows as described, avoid duplicates
            if row_type == "word" and token and url and token not in seen:
                seen.add(token)
                queue.append({"token": token, "url": url})
                
    if limit is not None:
        queue = queue[:limit]
        print(f"Limiting execution to {limit} entries for testing.")
    else:
        print(f"Found {len(queue)} unique word entries to process.")

    written_count = 0
    cached_count = 0
    api_count = 0
    start_time = time.time()
    url_cache = {}  # Maps raw_url -> {raw_url, api_url, definition}
    
    # Write to clean_data/definitions.csv
    print(f"Saving compiled definitions to {OUT_FILE}...")
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        f.write("word,raw_url,api_url,definition\n")
        
        for item in queue:
            # Check if we've already fetched this URL
            if item["url"] in url_cache:
                # Reuse cached result
                cached_data = url_cache[item["url"]]
                word = item["token"]
                raw_url = cached_data["raw_url"]
                api_url = cached_data["api_url"]
                definition = cached_data["definition"]
                cached_count += 1
            else:
                # Fetch from API
                data = fetch_word_definition_data(item["token"], item["url"])
                if data:
                    word = data["word"]
                    raw_url = data["raw_url"]
                    api_url = data["api_url"]
                    definition = data["definition"]
                    # Cache this result for future use
                    url_cache[item["url"]] = {
                        "raw_url": raw_url,
                        "api_url": api_url,
                        "definition": definition
                    }
                    api_count += 1
                    # Sleep 150ms per API call to not overwhelm the API server!
                    time.sleep(0.15)
                else:
                    continue
            
            # Write row with only definition quoted
            quoted_definition = f'"{definition}"' if definition else '""'
            f.write(f"{word},{raw_url},{api_url},{quoted_definition}\n")
            
            written_count += 1
            
            # Print periodic status metrics
            if written_count % 100 == 0 and written_count > 0:
                elapsed = time.time() - start_time
                print(f"Processed {written_count}/{len(queue)} items... ({(elapsed / written_count):.2f}s per iter, {api_count} API calls, {cached_count} cached)")
            
    print(f"\nExtraction complete! {written_count} words saved ({api_count} API queries, {cached_count} cached reuses).")

if __name__ == "__main__":
    test_limit = None
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_limit = 20  # Fast sample size limits execution during test checks
        
    extract_definitions(limit=test_limit)
