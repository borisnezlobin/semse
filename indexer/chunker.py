"""Shared chunking logic. Produces uniform Chunk objects across sources."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

import numpy as np

from contacts import ContactResolver, UNKNOWN_LABEL
from parse_imessage import IMessageRow

# Thread-aware iMessage chunking knobs. Adjust and re-index if the ambient
# bucket fragments too aggressively (raise threshold/min_seg) or fails to
# split clearly distinct conversations (lower threshold).
THREAD_CHUNK_CAP = 20         # max messages per chunk (both thread + ambient)
COHESION_WINDOW = 3           # past/future window size for TextTiling-style comparison
COHESION_THRESHOLD = 0.25     # split where past-vs-future window cosine drops below this
MIN_SEGMENT_LEN = 5           # don't emit ambient segments smaller than this (merge forward)


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


def _build_chunk_from_rows(
    rows: list[IMessageRow],
    title: str | None,
    resolver: ContactResolver | None,
) -> Chunk:
    """Build a single Chunk from a list of IMessageRow already sorted by date."""
    structured: list[ChunkMessage] = []
    for r in rows:
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
    return Chunk(
        chunk_id=_new_id(),
        source="imessage",
        contact_names=participants or ["Me"],
        date_start=rows[0].date_iso,
        date_end=rows[-1].date_iso,
        text=_format_imessage_block(structured),
        row_ids=[r.row_id for r in rows],
        messages=structured,
        chat_title=title,
    )


def _split_oversized(
    rows: list[IMessageRow],
    title: str | None,
    resolver: ContactResolver | None,
    cap: int = THREAD_CHUNK_CAP,
) -> Iterator[Chunk]:
    """Emit a single chunk if rows fit; otherwise sequential sub-chunks of size cap."""
    if not rows:
        return
    if len(rows) <= cap:
        yield _build_chunk_from_rows(rows, title, resolver)
        return
    for start in range(0, len(rows), cap):
        slab = rows[start : start + cap]
        if slab:
            yield _build_chunk_from_rows(slab, title, resolver)


def _segment_ambient(
    rows: list[IMessageRow],
    embeddings: np.ndarray,
    cap: int = THREAD_CHUNK_CAP,
    window: int = COHESION_WINDOW,
    threshold: float = COHESION_THRESHOLD,
    min_segment: int = MIN_SEGMENT_LEN,
) -> list[list[int]]:
    """TextTiling-style segmentation over per-message embeddings.

    At each candidate boundary position i, compare the mean embedding of the
    `window` messages immediately before i to the mean of the `window`
    messages starting at i. Where this past-vs-future cohesion drops below
    `threshold`, mark a segment boundary. Boundaries must be at least
    `min_segment` messages apart to avoid thrashing on short noisy iMessage
    chatter. Also splits hard at `cap` messages per segment.

    Returns segments as lists of indices into `rows`.
    """
    n = len(rows)
    if n == 0:
        return []
    if n <= window * 2:
        # Too short for meaningful window comparison; emit as a single segment
        # (with the hard cap applied as a fallback).
        if n <= cap:
            return [list(range(n))]
        return [list(range(i, min(i + cap, n))) for i in range(0, n, cap)]

    # Collect candidate boundary positions in date order, plus forced cuts at the cap.
    boundaries: list[int] = []
    last_boundary = 0
    for i in range(window, n - window):
        # Hard cap: if we'd exceed `cap` messages since the last boundary, cut here.
        if i - last_boundary >= cap:
            boundaries.append(i)
            last_boundary = i
            continue
        # Must be at least min_segment past the previous boundary.
        if i - last_boundary < min_segment:
            continue
        past = embeddings[i - window : i].mean(axis=0)
        future = embeddings[i : i + window].mean(axis=0)
        past_n = past / (np.linalg.norm(past) + 1e-8)
        future_n = future / (np.linalg.norm(future) + 1e-8)
        sim = float(np.dot(past_n, future_n))
        if sim < threshold:
            boundaries.append(i)
            last_boundary = i

    # Build segments. Also enforce the cap on the final segment.
    segments: list[list[int]] = []
    prev = 0
    for b in boundaries:
        segments.append(list(range(prev, b)))
        prev = b
    # Tail: split into cap-sized pieces if oversized.
    tail = list(range(prev, n))
    while len(tail) > cap:
        segments.append(tail[:cap])
        tail = tail[cap:]
    if tail:
        segments.append(tail)
    return segments


def chunk_imessages(
    rows: Iterable[IMessageRow],
    resolver: ContactResolver | None = None,
    embedder=None,
) -> Iterator[Chunk]:
    """Thread-aware chunker for iMessage.

    Within each chat_id (rows already arrive sorted by chat then date):
      1. Bucket by `thread_originator_guid`: each non-NULL value names a
         distinct explicit reply thread. The originator message itself joins
         the thread it spawned when its guid matches an originator-guid value.
      2. Everything else forms the "ambient" bucket — the chat's main flow.
      3. Thread buckets emit one chunk each (sub-split if > THREAD_CHUNK_CAP).
      4. The ambient bucket is segmented by embedding cohesion: each message
         is embedded; consecutive messages stay together while cosine to the
         running centroid stays above COHESION_THRESHOLD, then split. Cap at
         THREAD_CHUNK_CAP messages.
    """
    current_chat: int | None = None
    buffer: list[IMessageRow] = []
    chat_title: str | None = None

    def flush_chat(chat_buf: list[IMessageRow], title: str | None) -> Iterator[Chunk]:
        if not chat_buf:
            return

        # 1. Identify which guids appear as originator-guids inside this chat.
        thread_keys = {r.thread_originator_guid for r in chat_buf if r.thread_originator_guid}

        threads: dict[str, list[IMessageRow]] = {}
        ambient: list[IMessageRow] = []
        for r in chat_buf:
            key = r.thread_originator_guid
            if key:
                threads.setdefault(key, []).append(r)
            elif r.guid and r.guid in thread_keys:
                # Root message of a thread: group it with the thread it spawned.
                threads.setdefault(r.guid, []).append(r)
            else:
                ambient.append(r)

        # 2. Emit thread chunks, sorted by date.
        for key, rs in threads.items():
            rs.sort(key=lambda r: r.date_iso)
            yield from _split_oversized(rs, title, resolver)

        # 3. Segment the ambient bucket by embedding cohesion.
        if not ambient:
            return
        if embedder is None or len(ambient) == 1:
            # Without an embedder we fall back to fixed-size chunks of 20 in
            # date order. Single-message ambient is just one chunk.
            yield from _split_oversized(ambient, title, resolver)
            return

        texts = [r.text or "" for r in ambient]
        embeddings = embedder.embed_batch(texts)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        segments = _segment_ambient(ambient, embeddings)
        for seg in segments:
            seg_rows = [ambient[i] for i in seg]
            if seg_rows:
                yield from _split_oversized(seg_rows, title, resolver)

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
    chunks = list(chunk_imessages(fake))
    print(f"35 rows → {len(chunks)} chunks")
    for c in chunks:
        print(f"  {len(c.row_ids):2d} rows  contacts={c.contact_names}")
