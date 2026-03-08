import argparse
import json
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from app.core.config import settings

TAR_NAME = "20_newsgroups.tar.gz"


def load_from_local_zip(zip_path: Path) -> tuple[list[str], list[str]]:
    """
    Read all posts from the UCI-format zip file.
    Returns (texts, newsgroup_labels) as parallel lists.

    Streams tar.gz from inside the zip without extracting to disk.
    """
    print(f"Reading dataset from: {zip_path}")

    texts = []
    newsgroups = []
    skipped = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        tar_entries = [n for n in zf.namelist() if n.endswith(TAR_NAME)]
        if not tar_entries:
            raise FileNotFoundError(
                f"Could not find '{TAR_NAME}' inside {zip_path}.\n"
                f"Zip contents: {zf.namelist()}"
            )

        tar_entry = tar_entries[0]
        print(f"  Found: {tar_entry}")

        with zf.open(tar_entry) as tar_bytes:
            with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tf:
                members = tf.getmembers()
                file_members = [m for m in members if m.isfile()]
                print(f"  Total post files: {len(file_members)}\n")

                for member in tqdm(file_members, desc="Reading posts"):
                    # Path structure: 20_newsgroups/<newsgroup>/<post_id>
                    parts = Path(member.name).parts
                    if len(parts) < 3:
                        skipped += 1
                        continue

                    newsgroup = parts[1]  # e.g. "alt.atheism"

                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            skipped += 1
                            continue
                        raw_bytes = f.read()
                        # Newsgroup posts from 1993 aren't always UTF-8.
                        # latin-1 maps all 256 byte values — never raises.
                        try:
                            text = raw_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            text = raw_bytes.decode("latin-1")
                    except Exception:
                        skipped += 1
                        continue

                    texts.append(text)
                    newsgroups.append(newsgroup)

    print(f"\n  Loaded: {len(texts)} posts | Skipped (unreadable): {skipped}")
    return texts, newsgroups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Path to twenty+newsgroups.zip",
    )
    args = parser.parse_args()

    zip_path = args.zip

    # Auto-detect if --zip not passed
    if zip_path is None:
        candidates = [
            Path("twenty+newsgroups.zip"),
            Path("data/twenty+newsgroups.zip"),
            Path.home() / "Downloads" / "twenty+newsgroups.zip",
        ]
        for candidate in candidates:
            if candidate.exists():
                zip_path = candidate
                print(f"Auto-detected: {zip_path}")
                break

    if zip_path is None or not zip_path.exists():
        print("ERROR: No zip file found.")
        print("Pass it explicitly: python scripts/01_download_data.py --zip /path/to/file.zip")
        sys.exit(1)

    texts, newsgroup_labels = load_from_local_zip(zip_path)

    # ── Save to JSONL ────────────────────────────────────────────────────────
    # JSONL (one JSON object per line) is the right format here:
    #   - Streamable: 02_preprocess.py can process line-by-line without
    #     loading all 18k records into memory at once.
    #   - Inspectable: open in any text editor to audit individual posts.
    #   - Appendable: easy to add records without rewriting the whole file.
    raw_dir = settings.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "newsgroups_raw.jsonl"

    all_categories = sorted(set(newsgroup_labels))

    print(f"\nSaving to {raw_path}...")
    with open(raw_path, "w", encoding="utf-8") as f:
        for idx, (text, newsgroup) in enumerate(
            tqdm(zip(texts, newsgroup_labels), total=len(texts))
        ):
            record = {
                "id": idx,
                "text": text,
                "newsgroup": newsgroup,
                "target": all_categories.index(newsgroup),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(texts)} posts saved to {raw_path}")
    print(f"\nCategory distribution:")
    counts = Counter(newsgroup_labels)
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()