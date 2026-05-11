"""Parse Apple Mail using the Envelope Index sqlite DB and per-message .emlx files."""
from __future__ import annotations

import email
import glob
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from email import policy as email_policy
from email.message import Message
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup

from chunker import MailMessage

MAIL_ROOT_GLOB = str(Path.home() / "Library" / "Mail" / "V*")

NEWSLETTER_SENDER_PATTERNS = [
    re.compile(r"^no[-_.]?reply@", re.I),
    re.compile(r"^notifications?@", re.I),
    re.compile(r"^updates?@", re.I),
    re.compile(r"^newsletter@", re.I),
    re.compile(r"^marketing@", re.I),
    re.compile(r"^hello@", re.I),
    re.compile(r"^team@", re.I),
    re.compile(r"^info@", re.I),
    re.compile(r"^support@", re.I),
    re.compile(r"unsubscribe", re.I),
    re.compile(r"@mailer\.", re.I),
    re.compile(r"@email\.", re.I),
    re.compile(r"@news\.", re.I),
]


def _is_newsletter_sender(addr: str | None) -> bool:
    if not addr:
        return False
    return any(p.search(addr) for p in NEWSLETTER_SENDER_PATTERNS)


def _find_mail_root() -> Path:
    candidates = sorted(glob.glob(MAIL_ROOT_GLOB))
    if not candidates:
        raise FileNotFoundError(f"No Apple Mail data found at {MAIL_ROOT_GLOB}")
    return Path(candidates[-1])  # most recent versioned folder


def _envelope_index_path(mail_root: Path) -> Path:
    p = mail_root / "MailData" / "Envelope Index"
    if not p.exists():
        raise FileNotFoundError(f"Envelope Index not found at {p}")
    return p


def parse_emlx(path: Path) -> Message | None:
    """Parse an Apple .emlx file: skip leading byte count, strip trailing plist."""
    try:
        with open(path, "rb") as f:
            f.readline()
            raw = f.read()
    except OSError:
        return None
    xml_start = raw.find(b"<?xml")
    if xml_start != -1:
        raw = raw[:xml_start]
    try:
        return email.message_from_bytes(raw, policy=email_policy.default)
    except Exception:
        return None


def _extract_body(msg: Message) -> str:
    """Prefer text/plain; fall back to HTML stripped via BeautifulSoup."""
    plain_parts: list[str] = []
    html_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    plain_parts.append(part.get_content())
                except Exception:
                    pass
            elif ctype == "text/html":
                try:
                    html_parts.append(part.get_content())
                except Exception:
                    pass
    else:
        ctype = msg.get_content_type()
        try:
            content = msg.get_content()
        except Exception:
            content = ""
        if ctype == "text/plain":
            plain_parts.append(content)
        elif ctype == "text/html":
            html_parts.append(content)

    if plain_parts:
        return "\n\n".join(p.strip() for p in plain_parts if p)
    if html_parts:
        soup = BeautifulSoup("\n".join(html_parts), "lxml")
        for s in soup(["script", "style", "head"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        # Collapse runs of blank lines
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    return ""


def _ts_to_iso(ts) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def _detect_message_columns(conn: sqlite3.Connection) -> dict[str, str]:
    """Apple's `messages` table column names drift across macOS versions; auto-detect.

    Note: on macOS 14+, `messages.subject` is an INTEGER foreign key into a separate
    `subjects(ROWID, subject TEXT)` table. We always join through that.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(messages)")
    cols = {row[1].lower() for row in cur.fetchall()}
    pick = lambda *cands: next((c for c in cands if c in cols), None)
    return {
        "rowid": "ROWID",
        "sender": pick("sender", "sender_address") or "sender",
        "date_received": pick("date_received", "date_sent", "date_created") or "date_received",
        "thread_id": pick("conversation_id", "thread_id", "list_id_hash"),
        "mailbox": pick("mailbox"),
        "remote_id": pick("remote_id", "external_id"),
    }


def _build_emlx_index(mail_root: Path) -> dict[str, Path]:
    """Walk the mail tree once and map basename (without .emlx/.partial.emlx) → full path.

    Apple stores files like 1234.emlx and 1234.partial.emlx; we key on '1234'.
    Without this index, looking up each row via glob is O(files * rows), which dominates
    runtime — millions of stat calls for ~1k rows × ~1k files.
    """
    index: dict[str, Path] = {}
    for path in mail_root.rglob("*.emlx"):
        name = path.name
        if name.endswith(".partial.emlx"):
            key = name[:-len(".partial.emlx")]
        else:
            key = name[:-len(".emlx")]
        index.setdefault(key, path)
    return index


def _find_emlx_for_row(emlx_index: dict[str, Path], row_id: int, remote_id: str | None) -> Path | None:
    if remote_id is not None:
        hit = emlx_index.get(str(remote_id))
        if hit:
            return hit
    return emlx_index.get(str(row_id))


def iter_mail_threads(limit: int | None = None) -> Iterator[list[MailMessage]]:
    """Yield message-threads. Each thread is a list of MailMessage sorted by date."""
    mail_root = _find_mail_root()
    db_path = _envelope_index_path(mail_root)
    print(f"  building emlx file index from {mail_root}…")
    emlx_index = _build_emlx_index(mail_root)
    print(f"  indexed {len(emlx_index):,} emlx files")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cols = _detect_message_columns(conn)
        # Try to fetch sender display via the addresses table if it exists.
        sender_join = ""
        sender_select = f"m.{cols['sender']} AS sender_raw"
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='addresses'")
        if cur.fetchone():
            sender_join = "LEFT JOIN addresses a ON m.sender = a.ROWID"
            sender_select = "COALESCE(a.address, '') AS sender_raw, COALESCE(a.comment, '') AS sender_name"

        thread_col = cols["thread_id"] or cols["rowid"]
        date_col = cols["date_received"]
        remote_col = cols["remote_id"]

        # Subjects live in their own table; messages.subject is the FK.
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='subjects'")
        subj_select = (
            "COALESCE(s.subject, '') AS subject"
            if cur.fetchone()
            else "'' AS subject"
        )
        subj_join = "LEFT JOIN subjects s ON m.subject = s.ROWID"

        query = f"""
            SELECT
              m.ROWID AS row_id,
              m.{thread_col} AS thread_id,
              {subj_select},
              {sender_select},
              m.{date_col}   AS date_received
              {", m." + remote_col + " AS remote_id" if remote_col else ", NULL AS remote_id"}
            FROM messages m
            {sender_join}
            {subj_join}
            ORDER BY thread_id, date_received ASC
        """

        threads: dict[str, list[MailMessage]] = defaultdict(list)
        produced = 0
        for row in cur.execute(query):
            row_dict = dict(zip([d[0] for d in cur.description], row))
            sender_addr = row_dict.get("sender_raw") or ""
            if _is_newsletter_sender(sender_addr):
                continue
            emlx = _find_emlx_for_row(emlx_index, row_dict["row_id"], row_dict.get("remote_id"))
            if not emlx:
                continue
            parsed = parse_emlx(emlx)
            if not parsed:
                continue
            body = _extract_body(parsed)
            if len(body.strip()) < 20:
                continue
            recipients = _addr_list(parsed, "To") + _addr_list(parsed, "Cc")
            thread_key = str(row_dict["thread_id"] or row_dict["row_id"])
            threads[thread_key].append(
                MailMessage(
                    row_id=int(row_dict["row_id"]),
                    thread_id=thread_key,
                    subject=str(row_dict.get("subject") or "").strip(),
                    sender=sender_addr,
                    recipients=recipients,
                    date_iso=_ts_to_iso(row_dict.get("date_received")),
                    body=body,
                )
            )
            produced += 1
            if limit and produced >= limit:
                break
    finally:
        conn.close()

    for tid, msgs in threads.items():
        if msgs:
            yield msgs


def _addr_list(msg: Message, header: str) -> list[str]:
    raw = msg.get(header)
    if not raw:
        return []
    out: list[str] = []
    for piece in str(raw).split(","):
        piece = piece.strip()
        if piece:
            out.append(piece)
    return out


if __name__ == "__main__":
    n = 0
    for thread in iter_mail_threads(limit=20):
        n += 1
        print(f"thread {n}: {len(thread)} messages — {thread[0].subject[:60]}")
        if n >= 5:
            break
