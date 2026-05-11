"""Minimal extractor for Apple's typedstream-encoded `attributedBody` blobs.

iMessage stores ~40% of message text not in `message.text` but in
`message.attributedBody` — a binary NSArchiver typedstream containing an
NSAttributedString. The plain string lives inside, length-prefixed, after the
class metadata header.

We don't parse the full typedstream format (it's gnarly and most messages don't
need it). Instead we use a straightforward heuristic: locate the NSString class
marker, advance past the type byte (0x2b), then read Apple's variable-length
size prefix and decode that many bytes as UTF-8.
"""
from __future__ import annotations


_STRING_MARKER = b"NSString"


def extract_text(blob: bytes | memoryview | None) -> str | None:
    if not blob:
        return None
    data = bytes(blob)
    if b"streamtyped" not in data[:32]:
        return None
    idx = data.find(_STRING_MARKER)
    if idx == -1:
        return None
    pos = idx + len(_STRING_MARKER)
    # Skip past the class metadata terminator (\x86 \x84 \x01 ... varies).
    # The actual string content is preceded by a `+` (0x2b) type code in the
    # NSAttributedString backing store.
    plus = data.find(b"\x2b", pos)
    if plus == -1 or plus - pos > 32:
        # Fallback: scan a few bytes ahead for a printable run.
        return _fallback_scan(data[pos:])
    pos = plus + 1
    if pos >= len(data):
        return None

    length, pos = _read_length(data, pos)
    if length is None or length <= 0 or pos + length > len(data):
        return _fallback_scan(data[pos:])
    raw = data[pos : pos + length]
    # Apple sometimes prefixes the byte payload with a 1-byte encoding indicator
    # (0x00 = UTF-8). The length count includes that byte, so the visible string
    # is one byte shorter. Strip leading control bytes defensively.
    while raw and raw[0] < 0x09:
        raw = raw[1:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _fallback_scan(raw)
    return text or None


def _read_length(data: bytes, pos: int) -> tuple[int | None, int]:
    """Apple's typedstream uses a 1-byte length, with 0x81/0x82/0x83/0x84 marking
    that the actual length is in the following 1/2/3/4 bytes (little-endian)."""
    if pos >= len(data):
        return None, pos
    b = data[pos]
    pos += 1
    if b < 0x81:
        return b, pos
    n = b - 0x80
    if pos + n > len(data):
        return None, pos
    length = int.from_bytes(data[pos : pos + n], "little")
    return length, pos + n


def _fallback_scan(data: bytes) -> str | None:
    """When length-prefix parsing fails, grab the longest run of printable UTF-8.

    Heavily defensive — only used when the structured parse breaks (rare).
    """
    import re
    # Match printable UTF-8-like runs of >= 4 chars. This will match emoji,
    # punctuation, accented letters, etc. via the broad byte range.
    candidates = re.findall(rb"[\x20-\x7e\xc2-\xf4][\x20-\x7e\x80-\xff]{3,}", data)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            text = c.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not text:
            continue
        # Skip obvious typedstream class names that leak through.
        if text in {"NSDictionary", "NSNumber", "NSObject", "NSString", "NSArray",
                    "NSAttributedString", "streamtyped", "NSMutableString",
                    "NSMutableAttributedString", "NSMutableDictionary"}:
            continue
        return text
    return None


if __name__ == "__main__":
    # Quick smoke test against chat.db.
    import shutil, sqlite3, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from parse_imessage import copy_chat_db_to_temp, imessage_date_to_iso
    db, tmp_dir = copy_chat_db_to_temp()
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT date, attributedBody FROM message "
            "WHERE (text IS NULL OR length(trim(text))=0) AND attributedBody IS NOT NULL "
            "ORDER BY date DESC LIMIT 20"
        )
        for date_ns, body in rows:
            text = extract_text(body)
            print(f"{imessage_date_to_iso(date_ns)[:19]}  →  {text!r}")
        conn.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
