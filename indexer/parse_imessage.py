"""Parse iMessage chat.db and yield message rows grouped by chat for chunking."""
from __future__ import annotations

import datetime
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

MAC_EPOCH_OFFSET = 978307200  # seconds between Unix epoch and Mac epoch (2001-01-01 UTC)

CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

MESSAGE_QUERY = """
SELECT
  m.rowid,
  m.text,
  m.attributedBody,
  m.payload_data,
  m.date,
  m.is_from_me,
  h.id         AS contact_id,
  c.rowid      AS chat_id,
  c.display_name,
  c.chat_identifier
FROM message m
LEFT JOIN handle h         ON m.handle_id       = h.rowid
LEFT JOIN chat_message_join cmj ON m.rowid      = cmj.message_id
LEFT JOIN chat c           ON cmj.chat_id        = c.rowid
WHERE
  (
    (m.text IS NOT NULL AND length(trim(m.text)) > 0)
    OR m.attributedBody IS NOT NULL
    OR m.payload_data IS NOT NULL
  )
  -- associated_message_guid is set on tap-back reactions ("Loved", "Liked", etc).
  -- These add no semantic content and pollute retrieval; drop them entirely.
  AND m.associated_message_guid IS NULL
ORDER BY chat_id, m.date ASC
"""


@dataclass
class IMessageRow:
    row_id: int
    text: str
    date_iso: str
    is_from_me: bool
    contact_id: str | None
    chat_id: int | None
    display_name: str | None
    chat_identifier: str | None


def imessage_date_to_iso(ns: int) -> str:
    """Convert iMessage nanoseconds-since-2001 timestamp to ISO 8601 UTC."""
    if ns is None:
        return ""
    unix_ts = ns / 1_000_000_000 + MAC_EPOCH_OFFSET
    return datetime.datetime.utcfromtimestamp(unix_ts).isoformat()


_REACTION_PREFIXES = (
    "loved “", "liked “", "disliked “", "emphasized “",
    "questioned “", "laughed at “", "loved an image", "liked an image",
    "loved \"", "liked \"", "disliked \"", "emphasized \"", "questioned \"",
    "laughed at \"", "reacted ", "removed a ",
)


def _is_reaction_or_noise(text: str | None) -> bool:
    if not text:
        return True
    if text.startswith("�"):
        return True
    stripped = text.strip()
    if not stripped:
        return True
    # Single emoji / single tap-back glyph
    if len(stripped) == 1:
        return True
    # Apple's textual reaction fallbacks (these slip through even after the
    # associated_message_guid filter when the contact is on Android/SMS).
    lower = stripped.lower()
    if any(lower.startswith(p) for p in _REACTION_PREFIXES):
        return True
    # Pure object-replacement chars (￼) — attachment placeholders with no
    # accompanying text. Rendered as small squares in iMessage.
    cleaned = stripped.replace("￼", "").strip()
    if not cleaned:
        return True
    return False


def copy_chat_db_to_temp() -> str:
    """Copy chat.db (and -wal/-shm sidecar files) to a temp dir; return temp db path."""
    if not CHAT_DB_PATH.exists():
        raise FileNotFoundError(f"chat.db not found at {CHAT_DB_PATH}")
    tmp_dir = tempfile.mkdtemp(prefix="semse_imessage_")
    dest = Path(tmp_dir) / "chat.db"
    shutil.copy2(CHAT_DB_PATH, dest)
    for sidecar in ("chat.db-wal", "chat.db-shm"):
        src = CHAT_DB_PATH.parent / sidecar
        if src.exists():
            shutil.copy2(src, Path(tmp_dir) / sidecar)
    return str(dest)


def iter_messages(db_path: str | None = None) -> Iterator[IMessageRow]:
    """Yield every iMessage row, sorted by chat then date ASC.

    Messages with NULL `text` but populated `attributedBody` (typically SMS from
    short-codes, edited messages, and reactions with content) are also recovered
    by extracting the inner NSString from the typedstream blob.
    """
    from typedstream import extract_text  # local import to avoid circular deps
    from link_preview import extract as extract_link_preview
    path = db_path or copy_chat_db_to_temp()
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.cursor()
        cur.execute(MESSAGE_QUERY)
        for row in cur:
            (row_id, text, attributed_body, payload_data, date_ns, is_from_me,
             contact_id, chat_id, display_name, chat_identifier) = row
            if not text or not text.strip():
                text = extract_text(attributed_body) or ""
            # Append link-preview metadata so titles/descriptions are searchable.
            if payload_data:
                preview = extract_link_preview(payload_data)
                if preview:
                    preview_text = preview.as_text()
                    if preview_text:
                        text = f"{text}\n[link] {preview_text}".strip() if text else f"[link] {preview_text}"
            if _is_reaction_or_noise(text):
                continue
            yield IMessageRow(
                row_id=row_id,
                text=text,
                date_iso=imessage_date_to_iso(date_ns),
                is_from_me=bool(is_from_me),
                contact_id=contact_id,
                chat_id=chat_id,
                display_name=display_name,
                chat_identifier=chat_identifier,
            )
    finally:
        conn.close()


def chat_label(row: IMessageRow) -> str:
    """Best-effort human label for a chat: display_name → contact_id → chat_identifier."""
    return row.display_name or row.contact_id or row.chat_identifier or "Unknown"


if __name__ == "__main__":
    # Smoke test: print 10 sample messages.
    n = 0
    for msg in iter_messages():
        speaker = "Me" if msg.is_from_me else chat_label(msg)
        print(f"[{msg.date_iso}] chat={msg.chat_id} {speaker}: {msg.text[:80]}")
        n += 1
        if n >= 10:
            break
    print(f"\nPrinted {n} messages.")
