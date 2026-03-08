"""
Apply cleaning pipeline and filter the corpus.

Reads data/raw/newsgroups_raw.jsonl line by line (memory efficient),
cleans each post, drops those too short to be useful, and writes
data/processed_corpus.jsonl.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from app.core.config import settings
from app.utils.preprocessing import clean_post, is_useful


def main():
    raw_path = settings.DATA_DIR / "raw" / "newsgroups_raw.jsonl"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run 01_download_data.py first.")
        sys.exit(1)

    with open(raw_path) as f:
        total_raw = sum(1 for _ in f)

    print(f"Processing {total_raw} raw posts...\n")

    out_path = settings.PROCESSED_DATA_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    dropped_by_ng = defaultdict(int)
    kept_by_ng = defaultdict(int)

    with open(raw_path) as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_raw):
            record = json.loads(line)
            cleaned = clean_post(record["text"])

            if not is_useful(cleaned):
                dropped += 1
                dropped_by_ng[record["newsgroup"]] += 1
                continue

            fout.write(json.dumps({
                "id": record["id"],
                "text": cleaned,
                "newsgroup": record["newsgroup"],
                "target": record["target"],
                "original_length": len(record["text"]),
                "cleaned_length": len(cleaned),
            }, ensure_ascii=False) + "\n")

            kept += 1
            kept_by_ng[record["newsgroup"]] += 1

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"PREPROCESSING SUMMARY")
    print(f"{'='*55}")
    print(f"Total raw:   {total_raw:>6}")
    print(f"Kept:        {kept:>6}  ({kept/total_raw*100:.1f}%)")
    print(f"Dropped:     {dropped:>6}  ({dropped/total_raw*100:.1f}%)")
    print(f"\nOutput: {out_path}")

    print(f"\nRetention by newsgroup:")
    all_ngs = sorted(set(list(kept_by_ng) + list(dropped_by_ng)))
    for ng in all_ngs:
        k = kept_by_ng.get(ng, 0)
        d = dropped_by_ng.get(ng, 0)
        t = k + d
        print(f"  {ng:<40} {k:>4}/{t:<4} ({k/t*100:.0f}%)")


if __name__ == "__main__":
    main()