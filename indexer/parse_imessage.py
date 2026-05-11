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
  m.guid,
  m.text,
  m.attributedBody,
  m.payload_data,
  m.date,
  m.is_from_me,
  m.thread_originator_guid,
  m.reply_to_guid,
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
    # Reply-chain metadata. iMessage's explicit "reply to" feature populates
    # thread_originator_guid on every message in the thread (pointing back at
    # the root); reply_to_guid points at the immediate parent. We use these
    # to bucket parallel threads instead of mixing them via sliding windows.
    guid: str | None = None
    thread_originator_guid: str | None = None
    reply_to_guid: str | None = None


def imessage_date_to_iso(ns: int) -> str:
    """Convert iMessage nanoseconds-since-2001 timestamp to ISO 8601 UTC."""
    if ns is None:
        return ""
    unix_ts = ns / 1_000_000_000 + MAC_EPOCH_OFFSET
    return datetime.datetime.utcfromtimestamp(unix_ts).isoformat()


def iso_to_imessage_ns(iso: str) -> int:
    """Inverse of imessage_date_to_iso — ISO 8601 string back to nanos-since-2001."""
    dt = datetime.datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int((dt.timestamp() - MAC_EPOCH_OFFSET) * 1_000_000_000)


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


def copy_chat_db_to_temp() -> tuple[str, str]:
    """Copy chat.db (and -wal/-shm sidecar files) to a fresh temp dir.

    Returns `(db_path, tmp_dir)`. The caller MUST `shutil.rmtree(tmp_dir)` when
    done — chat.db is ~800 MB and these copies accumulate fast across runs.
    """
    if not CHAT_DB_PATH.exists():
        raise FileNotFoundError(f"chat.db not found at {CHAT_DB_PATH}")
    tmp_dir = tempfile.mkdtemp(prefix="semse_imessage_")
    try:
        dest = Path(tmp_dir) / "chat.db"
        shutil.copy2(CHAT_DB_PATH, dest)
        for sidecar in ("chat.db-wal", "chat.db-shm"):
            src = CHAT_DB_PATH.parent / sidecar
            if src.exists():
                shutil.copy2(src, Path(tmp_dir) / sidecar)
    except BaseException:
        # If the copy fails partway (e.g. ENOSPC), nuke the partial dir — we
        # never returned it to the caller so nobody else will clean it up.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return str(dest), tmp_dir


def iter_messages(db_path: str | None = None, since_iso: str | None = None) -> Iterator[IMessageRow]:
    """Yield every iMessage row, sorted by chat then date ASC.

    Messages with NULL `text` but populated `attributedBody` (typically SMS from
    short-codes, edited messages, and reactions with content) are also recovered
    by extracting the inner NSString from the typedstream blob.

    `since_iso` (when set) filters at SQL level — only rows strictly newer than
    the cutoff are returned. Used by the additive `--update` indexing path so we
    don't pay typedstream/link-preview cost on already-indexed messages.
    """
    from typedstream import extract_text  # local import to avoid circular deps
    from link_preview import extract as extract_link_preview
    tmp_dir: str | None = None
    if db_path:
        path = db_path
    else:
        path, tmp_dir = copy_chat_db_to_temp()
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.cursor()
        query = MESSAGE_QUERY
        if since_iso:
            since_ns = iso_to_imessage_ns(since_iso)
            query = query.replace(
                "ORDER BY chat_id, m.date ASC",
                f"AND m.date > {since_ns}\nORDER BY chat_id, m.date ASC",
            )
        cur.execute(query)
        for row in cur:
            (row_id, guid, text, attributed_body, payload_data, date_ns, is_from_me,
             thread_originator_guid, reply_to_guid,
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
                guid=guid,
                thread_originator_guid=thread_originator_guid,
                reply_to_guid=reply_to_guid,
            )
    finally:
        conn.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
