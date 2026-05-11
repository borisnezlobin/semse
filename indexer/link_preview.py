"""Extract link-preview metadata (title, URL, summary, sender) from
`message.payload_data` — a binary plist NSKeyedArchiver archive.

The format is NSKeyedArchiver, which serializes object graphs into a flat
`$objects` array indexed by `CF$UID` references. We don't reconstruct the
graph; instead we just scan the `$objects` array for non-class strings and
URL-like values, which is enough to capture all the human-readable preview
content (title, description, URL, site name).
"""
from __future__ import annotations

import plistlib
import re
from dataclasses import dataclass

URL_RE = re.compile(r"^[a-z]+://[^\s]+$", re.I)
CLASS_NAMES = {
    "$null", "NSDictionary", "NSObject", "NSString", "NSURL", "NSArray",
    "NSData", "NSDate", "NSNumber", "NSValue", "NSMutableDictionary",
    "NSMutableString", "NSMutableArray", "NSKeyedArchiver",
    "RichLink", "LPLinkMetadata", "LPiTunesStoreLinkMetadata",
    "LPImageMetadata", "LPVideoMetadata", "LPYouTubeLinkMetadata",
    "LPAppleMusicLinkMetadata", "LPLinkMetadataAttributes",
}


@dataclass
class LinkPreview:
    url: str | None
    title: str | None
    summary: str | None
    site_name: str | None

    def as_text(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.summary and self.summary != self.title:
            parts.append(self.summary)
        if self.site_name:
            parts.append(f"({self.site_name})")
        if self.url:
            parts.append(self.url)
        return " — ".join(parts) if parts else ""


def extract(payload: bytes | memoryview | None) -> LinkPreview | None:
    if not payload:
        return None
    data = bytes(payload)
    if not data.startswith(b"bplist00"):
        return None
    try:
        plist = plistlib.loads(data)
    except Exception:
        return None
    objects = plist.get("$objects") if isinstance(plist, dict) else None
    if not isinstance(objects, list):
        return None

    # Collect candidate strings from $objects, dropping class names and refs.
    strings: list[str] = []
    for item in objects:
        if isinstance(item, str):
            s = item.strip()
            if not s or s in CLASS_NAMES:
                continue
            # Skip Objective-C class spec like "{CF$UID(NSString)}"
            if s.startswith("$") or s.startswith("NS") and len(s) < 40 and s.isalpha():
                continue
            strings.append(s)
        elif isinstance(item, bytes):
            # Skip raw binary blobs (images, etc.)
            continue

    url = next((s for s in strings if URL_RE.match(s)), None)
    # Heuristic: title/summary tend to be the longer human-readable strings.
    # Site name tends to be short and alphabetic.
    non_url_strings = [s for s in strings if not URL_RE.match(s) and len(s) > 1]
    # Drop obvious junk (UUIDs, internal identifiers).
    cleaned = [s for s in non_url_strings if not _looks_like_internal_id(s)]
    cleaned.sort(key=len, reverse=True)

    title = cleaned[0] if cleaned else None
    summary = cleaned[1] if len(cleaned) > 1 and cleaned[1] != title else None
    # Site name: shortest plausible candidate that isn't title/summary.
    site = None
    for s in cleaned[2:]:
        if 2 <= len(s) <= 40 and " " not in s.strip() and s != title and s != summary:
            site = s
            break

    if not (url or title or summary):
        return None
    return LinkPreview(url=url, title=title, summary=summary, site_name=site)


def _looks_like_internal_id(s: str) -> bool:
    # 32-hex chars, UUIDs, very long base64-ish strings
    if len(s) >= 32 and all(c in "0123456789abcdef-" for c in s.lower()):
        return True
    if len(s) > 200:
        return True
    return False


if __name__ == "__main__":
    import sqlite3, sys
    sys.path.insert(0, ".")
    from parse_imessage import copy_chat_db_to_temp, imessage_date_to_iso
    db = copy_chat_db_to_temp()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT date, text, payload_data FROM message "
        "WHERE payload_data IS NOT NULL ORDER BY date DESC LIMIT 10"
    )
    for date_ns, text, payload in rows:
        lp = extract(payload)
        if lp:
            print(f"{imessage_date_to_iso(date_ns)[:19]}  text={(text or '')[:30]!r}")
            print(f"  → {lp.as_text()[:200]}")
