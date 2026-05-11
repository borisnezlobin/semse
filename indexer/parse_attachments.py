"""Enumerate iMessage image attachments with their parent-chat context.

Each yielded `ImageAttachment` carries:
  - the on-disk file path (resolved from chat.db's tilde-prefixed path)
  - the parent message context: chat title, sender, date, surrounding text
  - the contact name (resolved via AddressBook)

Used by the indexer to embed images into a CLIP FAISS index alongside text.
"""
from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from contacts import ContactResolver
from parse_imessage import imessage_date_to_iso, copy_chat_db_to_temp

ATTACHMENT_QUERY = """
SELECT
  a.ROWID                AS attachment_id,
  a.filename             AS path,
  a.mime_type            AS mime,
  m.ROWID                AS message_id,
  m.text                 AS msg_text,
  m.attributedBody       AS msg_attr,
  m.date                 AS msg_date,
  m.is_from_me           AS is_from_me,
  h.id                   AS sender_handle,
  c.display_name         AS chat_title,
  c.chat_identifier      AS chat_identifier,
  c.ROWID                AS chat_id
FROM attachment a
JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id
JOIN message m ON maj.message_id = m.ROWID
LEFT JOIN handle h ON m.handle_id = h.ROWID
LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
LEFT JOIN chat c ON cmj.chat_id = c.ROWID
WHERE a.filename IS NOT NULL
  AND a.mime_type LIKE 'image/%'
  AND a.hide_attachment = 0
ORDER BY m.date DESC
"""

# Apple stores paths with leading tilde; expand once per call.
HOME = str(Path.home())


@dataclass
class ImageAttachment:
    attachment_id: int
    path: Path
    mime: str
    message_id: int
    msg_text: str            # original message body (or extracted from attributedBody)
    date_iso: str
    sender_name: str         # resolved via AddressBook, "Me" if outgoing, "Unknown (•1234)" otherwise
    sender_known: bool       # False if sender's handle isn't in AddressBook
    contact_key: str | None  # photo lookup key for the sender
    chat_title: str | None   # group chat name, if any
    chat_id: int | None
    is_from_me: bool


def _expand(path: str | None) -> Path | None:
    if not path:
        return None
    if path.startswith("~"):
        return Path(HOME + path[1:])
    return Path(path)


def iter_image_attachments(
    db_path: str | None = None,
    resolver: ContactResolver | None = None,
) -> Iterator[ImageAttachment]:
    from typedstream import extract_text  # avoid circular
    tmp_dir: str | None = None
    if db_path:
        path = db_path
    else:
        path, tmp_dir = copy_chat_db_to_temp()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for row in conn.execute(ATTACHMENT_QUERY):
            (att_id, file_path, mime, msg_id, msg_text, msg_attr, msg_date,
             is_from_me, sender_handle, chat_title, chat_ident, chat_id) = row
            full = _expand(file_path)
            if not full or not full.exists():
                continue
            text = (msg_text or "").strip() or (extract_text(msg_attr) or "")
            is_me = bool(is_from_me)
            if is_me:
                sender_name = "Me"
                sender_known = True
                contact_key = None
            elif resolver:
                contact = resolver.resolve(sender_handle)
                if contact:
                    sender_name = contact.name
                    sender_known = True
                    contact_key = resolver.photo_key(sender_handle)
                else:
                    sender_name = resolver.display_for(sender_handle)
                    sender_known = False
                    contact_key = None
            else:
                sender_name = sender_handle or "Unknown"
                sender_known = False
                contact_key = None
            yield ImageAttachment(
                attachment_id=att_id,
                path=full,
                mime=mime,
                message_id=msg_id,
                msg_text=text,
                date_iso=imessage_date_to_iso(msg_date),
                sender_name=sender_name,
                sender_known=sender_known,
                contact_key=contact_key,
                chat_title=chat_title,
                chat_id=chat_id,
                is_from_me=is_me,
            )
    finally:
        conn.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    n = 0
    by_mime: dict[str, int] = {}
    for att in iter_image_attachments():
        n += 1
        by_mime[att.mime] = by_mime.get(att.mime, 0) + 1
        if n <= 5:
            print(f"  {att.date_iso[:19]} from={att.sender_name} mime={att.mime} {att.path.name}")
    print(f"\ntotal accessible images: {n}")
    for mime, count in sorted(by_mime.items(), key=lambda x: -x[1]):
        print(f"  {mime:30s} {count:,}")
