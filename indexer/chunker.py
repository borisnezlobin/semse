"""Shared chunking logic. Produces uniform Chunk objects across sources."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

from contacts import ContactResolver, UNKNOWN_LABEL
from parse_imessage import IMessageRow


@dataclass
class ChunkMessage:
    """One message inside a chunk, structured for UI rendering."""
    sender: str           # resolved display name, or "Me", or fallback "Unknown (•1234)"
    is_from_me: bool
    text: str
    date_iso: str
    contact_key: str | None  # photo lookup key (None if no photo)
    known: bool = True    # False if sender's handle isn't in user's AddressBook


@dataclass
class Chunk:
    chunk_id: str
    source: str
    contact_names: list[str]
    date_start: str
    date_end: str
    text: str                          # what gets embedded
    row_ids: list[int]
    messages: list[ChunkMessage] = field(default_factory=list)
    subject: str | None = None         # mail-only: thread subject for header
    chat_title: str | None = None      # imessage-only: group chat name when present


def _new_id() -> str:
    return uuid.uuid4().hex


def _resolve_speaker(row: IMessageRow, resolver: ContactResolver | None) -> tuple[str, str | None, bool]:
    """(display_name, contact_key_for_photo, is_known_contact) for this row's sender."""
    if row.is_from_me:
        return ("Me", None, True)
    handle = row.contact_id
    if resolver:
        contact = resolver.resolve(handle)
        if contact:
            return (contact.name, resolver.photo_key(handle), True)
        return (resolver.display_for(handle), None, False)
    return (handle or UNKNOWN_LABEL, None, False)


def _format_imessage_block(messages: list[ChunkMessage]) -> str:
    return "\n".join(f"{m.sender}: {m.text}" for m in messages)


def chunk_imessages(
    rows: Iterable[IMessageRow],
    window: int = 20,
    stride: int = 10,
    resolver: ContactResolver | None = None,
) -> Iterator[Chunk]:
    """Sliding window over messages, grouped by chat_id. Stride 10 over windows of 20."""
    current_chat: int | None = None
    buffer: list[IMessageRow] = []
    chat_title: str | None = None

    def flush_chat(chat_buf: list[IMessageRow], title: str | None) -> Iterator[Chunk]:
        if not chat_buf:
            return
        n = len(chat_buf)
        starts = list(range(0, max(1, n - window + 1), stride))
        if not starts:
            starts = [0]
        if starts[-1] + window < n:
            starts.append(max(0, n - window))
        seen_starts: set[int] = set()
        for s in starts:
            if s in seen_starts:
                continue
            seen_starts.add(s)
            window_rows = chat_buf[s : s + window]
            if not window_rows:
                continue
            structured: list[ChunkMessage] = []
            for r in window_rows:
                name, key, known = _resolve_speaker(r, resolver)
                structured.append(
                    ChunkMessage(
                        sender=name,
                        is_from_me=r.is_from_me,
                        text=r.text,
                        date_iso=r.date_iso,
                        contact_key=key,
                        known=known,
                    )
                )
            participants = sorted({m.sender for m in structured if not m.is_from_me})
            yield Chunk(
                chunk_id=_new_id(),
                source="imessage",
                contact_names=participants or ["Me"],
                date_start=window_rows[0].date_iso,
                date_end=window_rows[-1].date_iso,
                text=_format_imessage_block(structured),
                row_ids=[r.row_id for r in window_rows],
                messages=structured,
                chat_title=title,
            )

    for row in rows:
        if current_chat is None:
            current_chat = row.chat_id
            chat_title = row.display_name
        if row.chat_id != current_chat:
            yield from flush_chat(buffer, chat_title)
            buffer = []
            current_chat = row.chat_id
            chat_title = row.display_name
        buffer.append(row)

    yield from flush_chat(buffer, chat_title)


@dataclass
class MailMessage:
    row_id: int
    thread_id: str
    subject: str
    sender: str
    recipients: list[str]
    date_iso: str
    body: str


def _format_mail_block(messages: list[MailMessage]) -> str:
    parts: list[str] = []
    for m in messages:
        header = f"From: {m.sender}\nDate: {m.date_iso}\nSubject: {m.subject}"
        parts.append(f"{header}\n\n{m.body.strip()}")
    return "\n\n---\n\n".join(parts)


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_mail_threads(
    threads: Iterable[list[MailMessage]],
    max_tokens_per_chunk: int = 800,
    resolver: ContactResolver | None = None,
    user_emails: set[str] | None = None,
) -> Iterator[Chunk]:
    """One chunk per thread; split oversized threads into sequential sub-chunks."""
    for thread in threads:
        if not thread:
            continue
        thread = sorted(thread, key=lambda m: m.date_iso)
        text = _format_mail_block(thread)
        if _approx_token_count(text) <= max_tokens_per_chunk:
            yield _build_mail_chunk(thread, text, resolver, user_emails)
            continue
        current: list[MailMessage] = []
        current_tokens = 0
        for msg in thread:
            piece = _format_mail_block([msg])
            piece_tokens = _approx_token_count(piece)
            if current and current_tokens + piece_tokens > max_tokens_per_chunk:
                yield _build_mail_chunk(current, _format_mail_block(current), resolver, user_emails)
                current = []
                current_tokens = 0
            current.append(msg)
            current_tokens += piece_tokens
            if piece_tokens > max_tokens_per_chunk:
                yield _build_mail_chunk(current, _format_mail_block(current), resolver, user_emails)
                current = []
                current_tokens = 0
        if current:
            yield _build_mail_chunk(current, _format_mail_block(current), resolver, user_emails)


def _build_mail_chunk(
    messages: list[MailMessage],
    text: str,
    resolver: ContactResolver | None,
    user_emails: set[str] | None = None,
) -> Chunk:
    user_emails = user_emails or set()
    structured: list[ChunkMessage] = []
    senders: list[str] = []
    for m in messages:
        sender_lc = (m.sender or "").lower().strip()
        is_me = sender_lc in user_emails
        if is_me:
            name = "Me"
            key = None
            known = True
        else:
            contact = resolver.resolve(m.sender) if resolver else None
            name = contact.name if contact else (m.sender or "Unknown")
            key = resolver.photo_key(m.sender) if (resolver and contact) else None
            known = contact is not None
        structured.append(
            ChunkMessage(
                sender=name,
                is_from_me=is_me,
                text=m.body,
                date_iso=m.date_iso,
                contact_key=key,
                known=known,
            )
        )
        if name not in senders and not is_me:
            senders.append(name)
    return Chunk(
        chunk_id=_new_id(),
        source="mail",
        contact_names=senders or ["Unknown"],
        date_start=messages[0].date_iso,
        date_end=messages[-1].date_iso,
        text=text,
        row_ids=[m.row_id for m in messages],
        messages=structured,
        subject=messages[0].subject or None,
    )


def chunk_message_to_dict(m: ChunkMessage) -> dict:
    return asdict(m)


if __name__ == "__main__":
    fake = [
        IMessageRow(
            row_id=i,
            text=f"msg {i}",
            date_iso=f"2025-01-01T00:00:{i:02d}",
            is_from_me=(i % 2 == 0),
            contact_id="+15551234567",
            chat_id=1,
            display_name=None,
            chat_identifier="+15551234567",
        )
        for i in range(35)
    ]
    chunks = list(chunk_imessages(fake, window=20, stride=10))
    print(f"35 rows → {len(chunks)} chunks")
    for c in chunks:
        print(f"  {len(c.row_ids):2d} rows  contacts={c.contact_names}")
