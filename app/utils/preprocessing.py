"""
Text cleaning pipeline for the 20 Newsgroups corpus.

What we KEEP vs DISCARD and WHY?
  Headers:      DISCARD. Carry author/routing metadata, not semantic content.
                Including them makes embeddings cluster by poster identity
                rather than topic — directly harming retrieval and clustering.
                EXCEPTION: Subject line is kept — it's the most information-
                dense part of any post.

  Quote blocks: DISCARD. Lines starting with ">" re-introduce content from
                other documents (data leakage), inflate corpus size, and pull
                embeddings toward the original post rather than the reply.

  Signatures:   DISCARD. Everything after "-- \n" is contact info / disclaimers.
                Pure noise with zero semantic value.

  UUENCODED:    DISCARD. Binary attachments encoded as ASCII gibberish.
                Destroys TF-IDF weights and confuses the embedder.

  PGP blocks:   DISCARD. Same reason as UUENCODED.

  Short posts:  DISCARD if < 50 chars after cleaning. These are "me too"
                replies or moderation notices — not enough signal to embed.

  Numbers/punct: KEEP. "RS/6000", "C++", "TCP/IP" carry semantic weight
                 in this corpus. We don't aggressively strip them.
"""

import re
from typing import Optional


# ── Compiled regex patterns (pre-compiled for speed over 20k docs) ──────────

# Quoted reply lines: ">", ">>", "> >", "  > " etc.
_QUOTE_LINE_RE = re.compile(r"^[ \t]*>+[^\n]*\n?", re.MULTILINE)

# Email signature block — everything after "-- \n" or "--\n" at line start
_SIG_RE = re.compile(r"\n--[ \t]*\n.*", re.DOTALL)

# UUENCODED attachment blocks
_UUENCODE_RE = re.compile(
    r"begin \d{3} \S+\n.+?\nend\n?", re.DOTALL | re.IGNORECASE
)

# PGP blocks
_PGP_RE = re.compile(
    r"-----BEGIN PGP.*?-----END PGP[^\n]*-----\n?", re.DOTALL
)

# Collapse 3+ blank lines down to one
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Collapse multiple spaces/tabs
_SPACES_RE = re.compile(r"[ \t]{2,}")


def extract_subject(raw: str) -> Optional[str]:
    """
    Pull the Subject header before we strip all headers.
    Removes Re:/Fwd: prefixes — they add no semantic value.
    """
    m = re.search(r"^Subject:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    subject = m.group(1).strip()
    # Strip reply/forward prefixes
    subject = re.sub(r"^(Re|Fwd|Fw|SV|AW):\s*", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"^\[[\w\s]+\]\s*", "", subject)
    return subject if len(subject) > 3 else None


def clean_post(raw: str, max_chars: int = 10_000) -> str:
    """
    Full cleaning pipeline for a single newsgroup post.

    Steps:
      1. Extract subject line (before headers are stripped)
      2. Remove UUENCODED and PGP blobs
      3. Remove signature block
      4. Split on first blank line to isolate body from headers
      5. Remove quoted reply lines from body
      6. Collapse whitespace
      7. Prepend subject back (most semantically dense line)
      8. Truncate to max_chars
    """
    subject = extract_subject(raw)

    # Remove binary/crypto blobs first — they confuse later regexes
    text = _UUENCODE_RE.sub("", raw)
    text = _PGP_RE.sub("", text)

    # Remove signature
    text = _SIG_RE.sub("", text)

    # Split headers from body on first blank line (RFC-2822 standard)
    if "\n\n" in text:
        _, body = text.split("\n\n", 1)
    else:
        # Malformed post — just use the whole thing
        body = text

    # Remove quoted reply lines
    body = _QUOTE_LINE_RE.sub("", body)

    # Clean up whitespace
    body = _BLANK_LINES_RE.sub("\n\n", body)
    body = _SPACES_RE.sub(" ", body)
    body = body.strip()

    # Prepend subject — it's the single most useful semantic line
    if subject:
        body = f"{subject}\n\n{body}"

    # Truncate: all-MiniLM-L6-v2 has a 256 wordpiece token limit.
    # Feeding more than ~1500 chars is wasteful — the model truncates anyway.
    # We keep up to 10k for NMF/TF-IDF which handles full text.
    return body[:max_chars]


def is_useful(text: str, min_chars: int = 50) -> bool:
    """
    Filter out posts with insufficient semantic content.

    50 chars ≈ 8 words. Below this the embedding is dominated by
    padding noise and cluster assignment becomes unreliable.
    These are typically empty replies or moderation notices.
    """
    return len(text.strip()) >= min_chars