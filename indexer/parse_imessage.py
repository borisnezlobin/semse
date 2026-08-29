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

# chat.style values in chat.db: a 1:1 direct message vs a named/group thread.
ONE_ON_ONE_CHAT_STYLE = 45
GROUP_CHAT_STYLE = 43

# chat_message_join can attach the same message.rowid to more than one chat row —
# most commonly an iMessage chat and its SMS twin for the same contact — which made
# the naive join emit (and index) every message twice. ROW_NUMBER keeps exactly one
# row per message: prefer the 1:1 style over the group style (style 45 > 43), then the
# lowest chat rowid for a stable tie-break. We also surface c.style and c.guid so the
# caller can compute a canonical conversation key that collapses any such twins.
MESSAGE_QUERY = """
SELECT
  row_id, guid, text, attributedBody, payload_data, date, is_from_me,
  thread_originator_guid, reply_to_guid, contact_id,
  chat_id, style, chat_guid, display_name, chat_identifier
FROM (
  SELECT
    m.rowid      AS row_id,
    m.guid       AS guid,
    m.text       AS text,
    m.attributedBody AS attributedBody,
    m.payload_data   AS payload_data,
    m.date       AS date,
    m.is_from_me AS is_from_me,
    m.thread_originator_guid AS thread_originator_guid,
    m.reply_to_guid          AS reply_to_guid,
    h.id         AS contact_id,
    c.rowid      AS chat_id,
    c.style      AS style,
    c.guid       AS chat_guid,
    c.display_name    AS display_name,
    c.chat_identifier AS chat_identifier,
    ROW_NUMBER() OVER (
      PARTITION BY m.rowid ORDER BY c.style DESC, c.rowid
    ) AS rn
  FROM message m
  LEFT JOIN handle h         ON m.handle_id  = h.rowid
  LEFT JOIN chat_message_join cmj ON m.rowid = cmj.message_id
  LEFT JOIN chat c           ON cmj.chat_id  = c.rowid
  WHERE
    (
      (m.text IS NOT NULL AND length(trim(m.text)) > 0)
      OR m.attributedBody IS NOT NULL
      OR m.payload_data IS NOT NULL
    )
    -- associated_message_guid is set on tap-back reactions ("Loved", "Liked", etc).
    -- These add no semantic content and pollute retrieval; drop them entirely.
    AND m.associated_message_guid IS NULL
    {since_clause}
)
WHERE rn = 1
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
    # Canonical conversation identity. Unlike chat_id (which can differ between an
    # iMessage chat and its SMS twin for the same person), conversation_key is stable
    # per real conversation: the normalized handle for 1:1 chats, the chat guid for
    # groups. The chunker groups on this so twin chats collapse into one thread.
    conversation_key: str | None = None
    style: int | None = None
    chat_guid: str | None = None
    # Reply-chain metadata. iMessage's explicit "reply to" feature populates
    # thread_originator_guid on every message in the thread (pointing back at
    # the root); reply_to_guid points at the immediate parent. We use these
    # to bucket parallel threads instead of mixing them via sliding windows.
    guid: str | None = None
    thread_originator_guid: str | None = None
    reply_to_guid: str | None = None


def conversation_key(
    style: int | None,
    contact_id: str | None,
    chat_identifier: str | None,
    chat_guid: str | None,
) -> str:
    """Canonical key that collapses twin chat rows for the same conversation.

    Group chats (style 43) are keyed by their durable chat guid. 1:1 chats
    (style 45, and any other/unknown style) are keyed by the normalized handle so
    iMessage/SMS twins for one contact map to a single conversation.
    """
    from contacts import normalize_handle
    if style == GROUP_CHAT_STYLE:
        return chat_guid or chat_identifier or "unknown_group"
    normalized = normalize_handle(contact_id) or normalize_handle(chat_identifier)
    return normalized or chat_identifier or contact_id or "unknown"


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
    """Yield every iMessage row, sorted by conversation then date ASC.

    Rows are materialized and sorted by `(conversation_key, date_iso, row_id)` so the
    chunker sees each canonical conversation as one contiguous, chronological stream —
    even where twin chat rows would otherwise interleave the same messages.

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
        since_clause = ""
        if since_iso:
            since_clause = f"AND m.date > {iso_to_imessage_ns(since_iso)}"
        cur.execute(MESSAGE_QUERY.format(since_clause=since_clause))
        rows: list[IMessageRow] = []
        for row in cur:
            (row_id, guid, text, attributed_body, payload_data, date_ns, is_from_me,
             thread_originator_guid, reply_to_guid,
             contact_id, chat_id, style, chat_guid, display_name, chat_identifier) = row
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
            rows.append(IMessageRow(
                row_id=row_id,
                text=text,
                date_iso=imessage_date_to_iso(date_ns),
                is_from_me=bool(is_from_me),
                contact_id=contact_id,
                chat_id=chat_id,
                display_name=display_name,
                chat_identifier=chat_identifier,
                conversation_key=conversation_key(style, contact_id, chat_identifier, chat_guid),
                style=style,
                chat_guid=chat_guid,
                guid=guid,
                thread_originator_guid=thread_originator_guid,
                reply_to_guid=reply_to_guid,
            ))
    finally:
        conn.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    rows.sort(key=lambda r: (r.conversation_key or "", r.date_iso, r.row_id))
    yield from rows


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
