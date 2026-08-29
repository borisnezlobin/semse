"""The consolidation scan — extraction WITH memory.

Reads one thread chronologically in chunk-aligned blocks, carrying a bounded rolling
working-state document, and writes a bi-temporal knowledge store (entities, aliases,
claims, mentions, contexts) into memory.db. Because the model reads in order with state,
coreference and alias-coining resolve at the moment they happen.

Every block: assemble transcript -> demand-page known entities -> one LLM call ->
post-process in ONE SQLite transaction (verbatim-evidence verification, entity/alias
upserts, model-declared supersession, undeclared-collision review queue, mentions,
contexts, checkpoint). Snapshots every 10 blocks; scan_log telemetry per block.

State keys are always (source, row_id) + dates — never chunk_id — so all state survives a
full rebuild that regenerates chunk_ids.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import memory_prompts as prompts
import memory_store as store
from llm_backend import get_backend, get_captioner

METADATA_DB_PATH = store.DATA_DIR / "metadata.db"
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"


def _parse_cap(raw: str | None) -> int | None:
    """A claim cap of 'none'/'0'/'off' means no ceiling; anything else parses to an int."""
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in ("", "none", "0", "off", "-1"):
        return None
    try:
        return int(v)
    except ValueError:
        return None


# Block sizing is tunable via env: the plan targets >=250 msgs / ~6k transcript tokens,
# but an 8B-class local model consolidates a smaller block far more reliably. At 250 msgs
# the prompt fills the context window and generation shifts it, dropping whole output
# sections; ~100 msgs leaves headroom for output and yields clean entities+contexts+claims.
# SEMSE_BLOCK_MAX_MESSAGES lets a stronger backend restore the 250-msg spec.
BLOCK_MAX_MESSAGES = int(os.getenv("SEMSE_BLOCK_MAX_MESSAGES", "100"))
BLOCK_MAX_TOKENS = int(os.getenv("SEMSE_BLOCK_MAX_TOKENS", "6000"))
STATE_MAX_TOKENS = int(os.getenv("SEMSE_STATE_MAX_TOKENS", "1200"))
# Claim cap: a guardrail against runaway output, NOT a target. SEMSE_CLAIM_CAP='none' lifts
# it entirely and leans on the prompt's recognition gate + confidence to control noise. The
# old blunt ceiling of 10 truncated real facts on dense blocks. SEMSE_MAX_CLAIMS is the
# legacy name, still honored as a fallback.
CLAIM_CAP = _parse_cap(os.getenv("SEMSE_CLAIM_CAP", os.getenv("SEMSE_MAX_CLAIMS", "10")))
# Attachment captioning: off by default so the production text scan is unchanged; the
# experiment sets SEMSE_SCAN_CAPTIONS=1. Requires a captioner (OpenAI key) + chat.db.
CAPTIONS_ENABLED = os.getenv("SEMSE_SCAN_CAPTIONS", "0").lower() not in ("0", "off", "false", "")
SNAPSHOT_EVERY = 10
PAGE_TERM_LIMIT = 8


# --------------------------------------------------------------------------- #
# Reading a thread from metadata.db (read-only)                                #
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    chunk_id: str
    row_ids: list[int]
    messages: list[dict]
    date_start: str
    date_end: str

    @property
    def min_row(self) -> int:
        return min(self.row_ids) if self.row_ids else 0

    @property
    def max_row(self) -> int:
        return max(self.row_ids) if self.row_ids else 0


def read_thread_chunks(
    metadata_db_path: Path | str,
    thread_key: str,
    source: str = "imessage",
    after_iso: str | None = None,
) -> list[Chunk]:
    """Load a thread's chunks in chronological order, optionally resuming after a watermark.

    Chunks are disjoint and ordered by date_start (verified against the corpus). The
    watermark is a chunk's date_start; skipping date_start <= watermark resumes cleanly
    because microsecond timestamps make per-chunk starts effectively unique.
    """
    src = sqlite3.connect(f"file:{metadata_db_path}?mode=ro", uri=True)
    try:
        rows = src.execute(
            "SELECT chunk_id, row_ids, messages, date_start, date_end FROM chunks "
            "WHERE thread_key=? AND source=? ORDER BY date_start",
            (thread_key, source),
        ).fetchall()
    finally:
        src.close()
    out: list[Chunk] = []
    for chunk_id, row_ids_json, messages_json, ds, de in rows:
        if after_iso and ds <= after_iso:
            continue
        out.append(Chunk(
            chunk_id=chunk_id,
            row_ids=json.loads(row_ids_json or "[]"),
            messages=json.loads(messages_json or "[]"),
            date_start=ds,
            date_end=de,
        ))
    return out


def assemble_blocks(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Group whole chunks into blocks of >=250 msgs or ~6k transcript tokens.

    Chunk-aligned: a chunk is never split across blocks, so contexts and situations that
    the chunker kept together stay together.
    """
    blocks: list[list[Chunk]] = []
    cur: list[Chunk] = []
    cur_msgs = 0
    cur_tokens = 0
    for ch in chunks:
        line = prompts.format_transcript(ch.messages, ch.row_ids)
        tok = prompts.estimate_tokens(line)
        if cur and (cur_msgs >= BLOCK_MAX_MESSAGES or cur_tokens >= BLOCK_MAX_TOKENS):
            blocks.append(cur)
            cur, cur_msgs, cur_tokens = [], 0, 0
        cur.append(ch)
        cur_msgs += len(ch.messages)
        cur_tokens += tok
    if cur:
        blocks.append(cur)
    return blocks


# --------------------------------------------------------------------------- #
# Attachment captioning — fold image/video content into the transcript          #
# --------------------------------------------------------------------------- #

_ATTACHMENT_QUERY = """
SELECT maj.message_id, a.ROWID, a.filename, a.mime_type
FROM message_attachment_join maj
JOIN attachment a ON maj.attachment_id = a.ROWID
WHERE a.hide_attachment = 0
  AND (a.mime_type LIKE 'image/%' OR a.mime_type LIKE 'video/%')
  AND maj.message_id IN (__ROW_IDS__)
"""


def load_attachment_map(
    row_ids: list[int], chat_db_path: Path | str = CHAT_DB_PATH
) -> dict[int, list[tuple[int, str, str]]]:
    """Map message row_id -> [(attachment_id, on-disk path, mime)] for visual attachments.

    Reads chat.db read-only (never copies/writes it). The chunks store carries no attachment
    link, so this is the authoritative join — and it sees videos and HEIC that the CLIP
    `images` table drops. Only files that still exist on disk are returned.
    """
    if not row_ids or not Path(chat_db_path).exists():
        return {}
    home = str(Path.home())
    out: dict[int, list[tuple[int, str, str]]] = {}
    conn = sqlite3.connect(f"file:{chat_db_path}?mode=ro", uri=True)
    try:
        for i in range(0, len(row_ids), 800):  # keep the IN list under SQLite's variable cap
            batch = row_ids[i:i + 800]
            q = _ATTACHMENT_QUERY.replace("__ROW_IDS__", ",".join("?" * len(batch)))
            for msg_id, att_id, filename, mime in conn.execute(q, batch):
                if not filename:
                    continue
                path = home + filename[1:] if filename.startswith("~") else filename
                if not Path(path).exists():
                    continue
                out.setdefault(msg_id, []).append((att_id, path, mime or ""))
    finally:
        conn.close()
    return out


def resolve_captions(
    conn: sqlite3.Connection,
    captioner,
    source: str,
    row_ids: list[int],
    chat_db_path: Path | str = CHAT_DB_PATH,
    verbose: bool = False,
) -> dict[int, str]:
    """Caption every visual attachment on these rows, returning row_id -> caption tag(s).

    Captions are cached in memory.db by attachment_id (pixels never change), so this is a
    no-op API-wise on re-runs and across experiment cells that seed the same cache. A caption
    the model can't produce is cached as a permanent failure so it's never re-attempted.
    """
    att_map = load_attachment_map(row_ids, chat_db_path)
    captions: dict[int, str] = {}
    for rid, atts in att_map.items():
        tags: list[str] = []
        for att_id, path, mime in atts:
            cached = store.get_caption(conn, source, att_id)
            if cached is None:  # never attempted
                text = captioner.caption(path, mime) if captioner else None
                store.put_caption(conn, source, att_id, path, mime, text)
                conn.commit()
                cached = text if text else store.CAPTION_FAILED
                if verbose and text:
                    print(f"    captioned att {att_id}: {text}", flush=True)
            if cached and cached != store.CAPTION_FAILED:
                kind = "video" if (mime or "").startswith("video/") else "image"
                tags.append(f"[{kind}: {cached}]")
        if tags:
            captions[rid] = " ".join(tags)
    return captions


# --------------------------------------------------------------------------- #
# Rolling working-state with code-enforced LRU eviction                        #
# --------------------------------------------------------------------------- #

@dataclass
class RollingState:
    """The bounded working set carried between blocks — computed FROM the store.

    The local model reliably fills only the first few output sections, so rather than have
    it emit an updated_state document (which it drops), we recompute the state in code from
    the entities it just touched and their current claims. That is deterministic, grounded
    in the authoritative store (the plan's Risk-1 mitigation: store-as-ground-truth), and
    still honors the <=1.2k-token cap with LRU eviction and pinned participants.
    """
    entries: list[dict] = field(default_factory=list)  # {name, kind, aliases, notes, last_block}
    pinned_norms: set[str] = field(default_factory=set)

    def to_model_view(self) -> list[dict]:
        return [{"name": e["name"], "kind": e.get("kind"),
                 "aliases": e.get("aliases", []), "notes": e.get("notes", "")}
                for e in self.entries]

    def to_json(self) -> str:
        return json.dumps({"entries": self.entries,
                           "pinned": sorted(self.pinned_norms)}, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str | None) -> "RollingState":
        if not s:
            return cls()
        d = json.loads(s)
        return cls(entries=d.get("entries", []), pinned_norms=set(d.get("pinned", [])))

    def rebuild_from_store(
        self, conn: sqlite3.Connection, touched_ids: set[str], block_index: int
    ) -> None:
        """Refresh entries for entities touched this block; age out untouched ones under cap."""
        by_norm = {store.norm_surface(e["name"]): e for e in self.entries}
        for eid in touched_ids:
            summary = _entity_state_summary(conn, eid)
            if not summary:
                continue
            summary["last_block"] = block_index
            by_norm[store.norm_surface(summary["name"])] = summary
        self.entries = sorted(by_norm.values(), key=lambda e: -e.get("last_block", 0))
        self._enforce_cap()

    def _enforce_cap(self) -> None:
        while prompts.estimate_tokens(self.to_json()) > STATE_MAX_TOKENS:
            evictable = [i for i, e in enumerate(self.entries)
                         if store.norm_surface(e["name"]) not in self.pinned_norms]
            if not evictable:
                break
            victim = min(evictable, key=lambda i: self.entries[i]["last_block"])
            self.entries.pop(victim)

    def ensure_pinned(self, pinned: list[dict]) -> None:
        """Guarantee participant + ent_me stubs exist so the model always sees them."""
        present = {store.norm_surface(e["name"]) for e in self.entries}
        for p in pinned:
            n = store.norm_surface(p["name"])
            self.pinned_norms.add(n)
            if n not in present:
                self.entries.append({"name": p["name"], "kind": p.get("kind", "person"),
                                     "aliases": p.get("aliases", []), "notes": "",
                                     "last_block": 0})


# --------------------------------------------------------------------------- #
# Participants and demand-paging candidate terms                               #
# --------------------------------------------------------------------------- #

_CAP_SEQ = re.compile(r"[A-Z][\w']+(?:\s+[A-Z][\w']+){0,2}")
_POSSESSIVE = re.compile(r"([A-Za-z][\w']+)'s\s+([a-z][\w']+)")


def candidate_terms(transcript: str) -> list[str]:
    """Proper-noun-ish terms to demand-page known entities the block might reference."""
    counts: Counter[str] = Counter()
    for m in _CAP_SEQ.findall(transcript):
        if len(m) > 2:
            counts[m] += 1
    for owner, noun in _POSSESSIVE.findall(transcript):
        counts[noun] += 1
    common = {"Me", "Unknown", "The", "I", "You", "We", "Ok", "Okay", "Yeah", "Lol", "Http", "Https"}
    ranked = [t for t, _ in counts.most_common() if t not in common]
    return ranked[:PAGE_TERM_LIMIT]


def resolve_participants(
    conn: sqlite3.Connection,
    thread_key: str,
    chunks: list[Chunk],
    participant_names: list[str] | None = None,
) -> list[dict]:
    """Get-or-create an entity per distinct non-Me sender; return pinned records.

    Binds each sender label as a kind:handle alias and the display name as a kind:name
    alias so one person unifies across contact labels. ent_me is always pinned.
    """
    pinned = [{"entity_id": store.ENT_ME, "name": "Me", "kind": "person", "aliases": []}]
    seen: dict[str, str] = {}  # sender label -> entity_id

    def register(sender: str, contact_key: str | None, date_iso: str | None) -> None:
        if not sender or sender == "Me" or sender in seen:
            return
        norm = store.norm_surface(sender)
        eid = store.find_entity_by_norm(conn, norm)
        if not eid:
            eid = store.create_entity(conn, sender, "person", contact_key,
                                      first_seen=date_iso, last_seen=date_iso)
        seen[sender] = eid
        if contact_key:
            store.set_entity_contact_key(conn, eid, contact_key)
        store.upsert_alias(conn, eid, sender, "handle", date_iso=date_iso, source_thread=thread_key)
        store.upsert_alias(conn, eid, sender, "name", date_iso=date_iso, source_thread=thread_key)

    if participant_names is not None:  # fixture path — no message contact keys
        for name in participant_names:
            register(name, None, None)
    else:
        for ch in chunks:
            for msg in ch.messages:
                if not msg.get("is_from_me"):
                    register(msg.get("sender"), msg.get("contact_key"), msg.get("date_iso"))

    for sender, eid in seen.items():
        pinned.append({"entity_id": eid, "name": sender, "kind": "person",
                       "aliases": [{"surface": sender, "kind": "name"}]})
    return pinned


# --------------------------------------------------------------------------- #
# Per-block post-processing (one transaction)                                  #
# --------------------------------------------------------------------------- #

@dataclass
class BlockResult:
    n_entities: int = 0
    n_claims: int = 0
    n_mentions: int = 0
    evidence_dropped: int = 0
    claims_dropped: int = 0
    review_queued: int = 0
    touched_ids: set[str] = field(default_factory=set)


def _entity_state_summary(conn: sqlite3.Connection, entity_id: str) -> dict | None:
    """Compact working-state record for one entity: name, aliases, latest valid claims."""
    ent = conn.execute(
        "SELECT canonical_name, kind FROM entities WHERE entity_id=? AND merged_into IS NULL",
        (entity_id,)).fetchone()
    if not ent:
        return None
    aliases = [r[0] for r in conn.execute(
        "SELECT DISTINCT surface FROM aliases WHERE entity_id=? AND kind IN "
        "('name','nickname') LIMIT 4", (entity_id,))]
    facts = [f"{r[0]}: {r[1]}" for r in conn.execute(
        "SELECT relation, object_text FROM claims WHERE subject_id=? AND invalidated_at IS NULL "
        "ORDER BY stated_at DESC LIMIT 3", (entity_id,))]
    return {"name": ent[0], "kind": ent[1], "aliases": aliases,
            "notes": "; ".join(facts)[:220]}


class _NameResolver:
    """Normalized name -> entity_id for subjects, speakers, objects, context entities."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.map: dict[str, str] = {"me": store.ENT_ME, "i": store.ENT_ME}

    def add(self, name: str, entity_id: str) -> None:
        n = store.norm_surface(name)
        if n:
            self.map.setdefault(n, entity_id)

    def resolve(self, name: str | None) -> str | None:
        if not name:
            return None
        n = store.norm_surface(name)
        if n in self.map:
            return self.map[n]
        eid = store.find_entity_by_norm(self.conn, n)
        if eid:
            self.map[n] = eid
        return eid


def _build_row_text_map(
    block: list[Chunk], captions: dict[int, str] | None = None
) -> dict[int, str]:
    """Row -> quote-validation haystack. When a row carries a caption, both the caption
    text and the original words are in-haystack, so the model may quote either as evidence."""
    row_text: dict[int, str] = {}
    for ch in block:
        for rid, msg in zip(ch.row_ids, ch.messages):
            text = msg.get("text") or ""
            if captions and rid in captions:
                # Validate against exactly what the model read: the caption folded in and the
                # ￼ placeholder stripped. Otherwise a quote spanning the caption boundary fails.
                text = prompts.apply_caption(text, captions[rid])
            row_text[rid] = text
    return row_text


def process_block(
    conn: sqlite3.Connection,
    output: dict,
    block: list[Chunk],
    *,
    thread_key: str,
    source: str,
    pinned: list[dict],
    paged: list[dict],
    scan_run_id: str,
    captions: dict[int, str] | None = None,
) -> BlockResult:
    """Apply one block's model output to the store. The caller wraps this together with
    the checkpoint/telemetry writes in ONE transaction so a block commits exactly-once:
    either its data AND its watermark land together, or neither does."""
    res = BlockResult()
    row_text = _build_row_text_map(block, captions)
    block_dates = [d for ch in block for d in (ch.date_start, ch.date_end) if d]
    block_first = min(block_dates) if block_dates else None
    block_last = max(block_dates) if block_dates else None

    # Caller holds the block transaction (data + checkpoint commit atomically).
    resolver = _NameResolver(conn)
    for p in pinned:
        resolver.add(p["name"], p["entity_id"])
    for pe in paged:
        resolver.add(pe["canonical_name"], pe["entity_id"])
        for a in pe.get("aliases", []):
            resolver.add(a["surface"], pe["entity_id"])

    # 1. Entities + aliases
    for ent in output.get("entities", []) or []:
        name = (ent.get("canonical_name") or "").strip()
        if not name:
            continue
        kind = (ent.get("kind") or "thing").strip()
        eid = resolver.resolve(name)
        if not eid:
            eid = store.create_entity(conn, name, kind, first_seen=block_first,
                                      last_seen=block_last)
            res.n_entities += 1
        resolver.add(name, eid)
        res.touched_ids.add(eid)
        store.touch_entity_seen(conn, eid, block_first)
        store.touch_entity_seen(conn, eid, block_last)
        store.upsert_alias(conn, eid, name, "name", date_iso=block_last,
                           source_thread=thread_key)
        for a in ent.get("aliases", []) or []:
            surface = (a.get("surface") or "").strip()
            if not surface:
                continue
            a_kind = (a.get("kind") or "nickname").strip()
            speaker = resolver.resolve(a.get("speaker")) if a.get("speaker") else None
            store.upsert_alias(conn, eid, surface, a_kind, speaker=speaker,
                               date_iso=block_last, valid_from=a.get("valid_from"),
                               valid_until=a.get("valid_until"), source_thread=thread_key)
            resolver.add(surface, eid)

    # 2. Claims (cap enforced; verbatim-evidence-or-drop; supersession/collision)
    claims_in = output.get("claims", []) or []
    if CLAIM_CAP is not None:
        claims_in = claims_in[:CLAIM_CAP]
    for claim in claims_in:
        subject = (claim.get("subject") or "").strip()
        relation = (claim.get("relation") or "").strip()
        if not subject or not relation:
            continue
        good_ev, dropped = _verify_evidence(claim.get("evidence"), row_text, source)
        res.evidence_dropped += dropped
        if not good_ev:
            res.claims_dropped += 1
            continue
        subject_id = resolver.resolve(subject)
        asserted_by = (claim.get("asserted_by") or "").strip() or None
        asserted_by_id = resolver.resolve(asserted_by)
        object_ref = claim.get("object_ref")
        object_id = resolver.resolve(object_ref) if object_ref else None
        if subject_id:
            res.touched_ids.add(subject_id)
        if object_id:
            res.touched_ids.add(object_id)
        stated_at = _iso_date(claim.get("stated_at")) or block_last
        valid_until = _iso_date(claim.get("valid_until"))

        prior = None
        if subject_id or subject:
            prior = store.find_supersedable_claim(conn, subject_id, subject, relation)

        claim_id = store.insert_claim(
            conn,
            subject_id=subject_id, subject_text=subject, relation=relation,
            object_text=(claim.get("object_text") or "").strip(),
            object_id=object_id, asserted_by=asserted_by, asserted_by_id=asserted_by_id,
            stated_at=stated_at, valid_from=_iso_date(claim.get("valid_from")),
            valid_until=valid_until, confidence=_as_float(claim.get("confidence")),
            derived=bool(claim.get("derived")), evidence=good_ev,
            thread_key=thread_key, scan_run_id=scan_run_id,
        )
        res.n_claims += 1

        if prior and prior["claim_id"] != claim_id:
            same_object = store.norm_surface(prior["object_text"] or "") == \
                store.norm_surface(claim.get("object_text") or "")
            if claim.get("supersedes_prior"):
                store.supersede_claim(conn, prior["claim_id"], claim_id,
                                      invalidated_at=stated_at, valid_until=valid_until or stated_at)
            elif not same_object:
                store.add_to_review_queue(conn, thread_key, subject_id, subject, relation,
                                          prior["claim_id"], claim_id,
                                          "undeclared same-subject collision")
                res.review_queued += 1

        # Mentions from claim participants at their evidence rows.
        for ev in good_ev:
            rid = ev["row_id"]
            date_iso = _row_date(block, rid)
            if subject_id:
                store.insert_entity_mention(conn, subject_id, source, rid, date_iso)
                res.n_mentions += 1
            if object_id:
                store.insert_entity_mention(conn, object_id, source, rid, date_iso)
                res.n_mentions += 1

    # 3. Contexts + per-chunk mentions
    for ctx in output.get("contexts", []) or []:
        r_start = _as_int(ctx.get("row_start"))
        r_end = _as_int(ctx.get("row_end"))
        if r_start is None or r_end is None:
            continue
        if r_start > r_end:
            r_start, r_end = r_end, r_start
        ent_ids: list[str] = []
        for nm in ctx.get("entities", []) or []:
            eid = resolver.resolve(nm)
            if eid and eid not in ent_ids:
                ent_ids.append(eid)
                res.touched_ids.add(eid)
        overlapping = [ch for ch in block
                       if ch.min_row <= r_end and ch.max_row >= r_start]
        span_dates = [ch.date_start for ch in overlapping] or [block_first]
        store.insert_chunk_context(
            conn, thread_key=thread_key, source=source, row_start=r_start, row_end=r_end,
            date_start=min(span_dates) if span_dates and span_dates[0] else block_first,
            date_end=max([ch.date_end for ch in overlapping] or [block_last]),
            context=(ctx.get("context") or "").strip(), entity_ids=ent_ids,
            scan_run_id=scan_run_id,
        )
        for ch in overlapping:
            for eid in ent_ids:
                store.insert_entity_mention(conn, eid, source, ch.min_row, ch.date_start)
                res.n_mentions += 1

    return res


def _verify_evidence(evidence, row_text: dict[int, str], source: str):
    good: list[dict] = []
    dropped = 0
    for ev in evidence or []:
        rid = _as_int(ev.get("row_id"))
        quote = ev.get("quote") or ""
        if rid is None or rid not in row_text or not store.verify_quote(quote, row_text[rid]):
            dropped += 1
            continue
        good.append({"source": source, "row_id": rid, "quote": quote})
    return good, dropped


def _row_date(block: list[Chunk], row_id: int) -> str | None:
    for ch in block:
        for rid, msg in zip(ch.row_ids, ch.messages):
            if rid == row_id:
                return msg.get("date_iso")
    return None


def _iso_date(s) -> str | None:
    if not s or not isinstance(s, str):
        return None
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    return m.group(0) if m else None


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Thread scan driver                                                           #
# --------------------------------------------------------------------------- #

def scan_thread(
    conn: sqlite3.Connection,
    thread_key: str,
    *,
    source: str = "imessage",
    metadata_db_path: Path | str = METADATA_DB_PATH,
    max_blocks: int | None = None,
    backend=None,
    captioner="auto",
    chat_db_path: Path | str = CHAT_DB_PATH,
    verbose: bool = True,
    _chunks: list[Chunk] | None = None,
    _participant_names: list[str] | None = None,
) -> dict:
    """Scan one thread from its checkpoint to the end (or max_blocks). Returns a summary."""
    backend = backend or get_backend()
    if captioner == "auto":
        captioner = get_captioner() if CAPTIONS_ENABLED else None
    system_prompt = prompts.build_system_prompt(CLAIM_CAP)
    scan_run_id = uuid.uuid4().hex

    checkpoint = store.get_checkpoint(conn, thread_key)
    after_iso = checkpoint["watermark_iso"] if checkpoint else None
    blocks_done = checkpoint["blocks_done"] if checkpoint else 0
    state = RollingState.from_json(checkpoint["state_json"] if checkpoint else None)

    chunks = _chunks if _chunks is not None else read_thread_chunks(
        metadata_db_path, thread_key, source, after_iso)
    if not chunks:
        return {"thread_key": thread_key, "blocks": 0, "note": "nothing to scan"}

    pinned = resolve_participants(conn, thread_key, chunks, _participant_names)
    conn.commit()
    state.ensure_pinned(pinned)

    all_blocks = assemble_blocks(chunks)
    blocks = all_blocks[:max_blocks] if max_blocks is not None else all_blocks

    # Caption every visual attachment on the rows we're about to scan (cached by
    # attachment_id, so cheap on re-runs). Empty when captioning is disabled or unavailable.
    captions: dict[int, str] = {}
    if captioner is not None:
        scan_row_ids = [r for block in blocks for ch in block for r in ch.row_ids]
        captions = resolve_captions(conn, captioner, source, scan_row_ids, chat_db_path,
                                    verbose=verbose)
        if verbose and captions:
            print(f"  captioned {len(captions)} message(s) with attachments", flush=True)

    summary = {"thread_key": thread_key, "blocks": 0, "entities": 0, "claims": 0,
               "mentions": 0, "evidence_dropped": 0, "claims_dropped": 0,
               "review_queued": 0, "captioned_msgs": len(captions), "per_block": []}

    for i, block in enumerate(blocks):
        block_index = blocks_done + i
        messages = [m for ch in block for m in ch.messages]
        row_ids = [r for ch in block for r in ch.row_ids]
        transcript = prompts.format_transcript(messages, row_ids, captions)
        today = max((m.get("date_iso") or "" for m in messages), default="")[:10]

        terms = candidate_terms(transcript)
        paged = store.page_entities(conn, terms, [p["entity_id"] for p in pinned])
        user_prompt = prompts.build_user_prompt(
            today_in_scan=today,
            participants=[p["name"] for p in pinned],
            working_state=state.to_model_view(),
            paged_entities=paged,
            transcript=transcript,
        )

        completion = backend.complete_verbose(
            system_prompt, user_prompt, prompts.OUTPUT_SCHEMA)
        output = completion.data

        last_chunk = block[-1]
        # ONE transaction per block: data + rolling state + watermark + telemetry land
        # together, or roll back together — an interrupt can never half-commit a block.
        with conn:
            res = process_block(conn, output, block, thread_key=thread_key, source=source,
                                pinned=pinned, paged=paged, scan_run_id=scan_run_id,
                                captions=captions)
            state.rebuild_from_store(conn, res.touched_ids, block_index)
            store.upsert_checkpoint(conn, thread_key, watermark_iso=last_chunk.date_start,
                                    watermark_row_id=last_chunk.max_row,
                                    blocks_done=block_index + 1, state_json=state.to_json(),
                                    status="running")
            if (block_index + 1) % SNAPSHOT_EVERY == 0:
                store.insert_snapshot(conn, thread_key, block_index + 1, state.to_json())
            store.insert_scan_log(
                conn, scan_run_id=scan_run_id, thread_key=thread_key, block_index=block_index,
                model=completion.model, tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out, n_messages=len(messages),
                n_entities=res.n_entities, n_claims=res.n_claims, n_mentions=res.n_mentions,
                evidence_dropped=res.evidence_dropped, claims_dropped=res.claims_dropped,
                review_queued=res.review_queued, duration_s=completion.duration_s)

        summary["blocks"] += 1
        summary["entities"] += res.n_entities
        summary["claims"] += res.n_claims
        summary["mentions"] += res.n_mentions
        summary["evidence_dropped"] += res.evidence_dropped
        summary["claims_dropped"] += res.claims_dropped
        summary["review_queued"] += res.review_queued
        summary["per_block"].append({
            "block": block_index, "msgs": len(messages), "dur_s": round(completion.duration_s, 1),
            "tok_in": completion.tokens_in, "tok_out": completion.tokens_out,
            "entities": res.n_entities, "claims": res.n_claims,
            "ev_dropped": res.evidence_dropped, "claims_dropped": res.claims_dropped})
        if verbose:
            print(f"  block {block_index} | {len(messages)} msgs | {completion.duration_s:.1f}s "
                  f"| in={completion.tokens_in} out={completion.tokens_out} "
                  f"| +{res.n_entities}e +{res.n_claims}c drop={res.evidence_dropped}ev/"
                  f"{res.claims_dropped}c rq={res.review_queued}", flush=True)

    done = len(blocks) >= len(all_blocks)  # consumed every block available after the watermark
    with conn:
        store.upsert_checkpoint(conn, thread_key,
                                watermark_iso=blocks[-1][-1].date_start if blocks else after_iso,
                                watermark_row_id=blocks[-1][-1].max_row if blocks else None,
                                blocks_done=blocks_done + len(blocks),
                                state_json=state.to_json(),
                                status="done" if done else "running")
    return summary


# --------------------------------------------------------------------------- #
# Fixture mode — synthetic arcs through the real pipeline                      #
# --------------------------------------------------------------------------- #

def _fixture_arcs(fixture: dict):
    """Yield (arc_name, messages, expectations) for each arc a fixture defines."""
    if "messages" in fixture:
        yield fixture.get("arc", "arc"), fixture["messages"], fixture.get("expectations", {})
        return
    for key in ("no_claim_arc", "uptake_arc"):
        if key in fixture:
            arc = fixture[key]
            yield key, arc["messages"], arc.get("expectations", {})


def _messages_to_chunks(messages: list[dict], base_row: int = 100000, per_chunk: int = 4):
    """Turn fixture messages into disjoint pseudo-chunks with synthetic row_ids."""
    chunks: list[Chunk] = []
    for start in range(0, len(messages), per_chunk):
        group = messages[start:start + per_chunk]
        row_ids = [base_row + start + i for i in range(len(group))]
        msgs = [dict(m) for m in group]
        for rid, m in zip(row_ids, msgs):
            m["row_id"] = rid
        chunks.append(Chunk(
            chunk_id=uuid.uuid4().hex, row_ids=row_ids, messages=msgs,
            date_start=group[0].get("date_iso", ""), date_end=group[-1].get("date_iso", "")))
    return chunks


def run_fixture(fixture_path: Path | str, memory_db_path: Path | str, backend=None) -> list[dict]:
    """Run each arc of a fixture as a pseudo-thread and check its expectations."""
    fixture = json.loads(Path(fixture_path).read_text())
    backend = backend or get_backend()
    results = []
    for arc_name, messages, expectations in _fixture_arcs(fixture):
        thread_key = f"{fixture['thread_key']}::{arc_name}"
        conn = store.connect(memory_db_path)
        try:
            chunks = _messages_to_chunks(messages)
            scan_thread(conn, thread_key, source="imessage", backend=backend,
                        verbose=False, _chunks=chunks,
                        _participant_names=fixture.get("participants"))
            check = check_expectations(conn, thread_key, fixture["arc"], arc_name, expectations)
        finally:
            conn.close()
        results.append({"fixture": fixture["arc"], "arc": arc_name, **check})
    return results


def check_expectations(
    conn: sqlite3.Connection, thread_key: str, arc: str, arc_name: str, expectations: dict
) -> dict:
    """Assert the by-construction ground truth for each fixture arc. Returns pass/detail."""
    claims = conn.execute(
        "SELECT subject_id, subject_text, relation, object_text, valid_until, stated_at, "
        "superseded_by, invalidated_at, derived, confidence FROM claims WHERE thread_key=?",
        (thread_key,)).fetchall()

    def any_claim(pred) -> bool:
        return any(pred(c) for c in claims)

    checks: list[tuple[str, bool]] = []

    if arc == "alias_drift":
        ent = conn.execute(
            "SELECT entity_id FROM entities WHERE lower(canonical_name) LIKE '%quillstream%' "
            "AND merged_into IS NULL").fetchone()
        checks.append(("entity Quillstream created", ent is not None))
        if ent:
            als = conn.execute("SELECT surface, kind FROM aliases WHERE entity_id=?",
                               (ent[0],)).fetchall()
            surfaces = " || ".join((a[0] or "").lower() for a in als)
            checks.append(("bound a descriptive alias (digest/cli/email)",
                           any(k in surfaces for k in ("digest", "cli", "email"))))
            checks.append(("bound the name alias Quillstream", "quillstream" in surfaces))
        checks.append(("pricing claim captured",
                       any_claim(lambda c: "5" in (c["object_text"] or "") or
                                 "pric" in (c["relation"] or "").lower())))

    elif arc == "expiry":
        checks.append(("bounded located_in claim with valid_until",
                       any_claim(lambda c: "lisbon" in (c["object_text"] or "").lower()
                                 and c["valid_until"] is not None)))
        bound = [c["valid_until"] for c in claims
                 if "lisbon" in (c["object_text"] or "").lower() and c["valid_until"]]
        checks.append(("valid_until resolved to ~2026-03-13",
                       any(b and b.startswith("2026-03-13") for b in bound)))

    elif arc == "supersession":
        sched = [c for c in claims if "sync" in (c["subject_text"] or "").lower()
                 and "schedul" in (c["relation"] or "").lower()]
        superseded = [c for c in sched if c["superseded_by"] or c["invalidated_at"]]
        current = [c for c in sched if not c["superseded_by"] and not c["invalidated_at"]]
        checks.append(("earlier sync time superseded", len(superseded) >= 1))
        checks.append(("exactly one current sync time",
                       len(current) == 1 and (not superseded or
                       current[0]["stated_at"] >= superseded[-1]["stated_at"])))

    elif arc == "lyrics_quotation":
        if arc_name == "no_claim_arc":
            checks.append(("no claim from the quoted lyric", len(claims) == 0))
        else:
            checks.append(("uptake produced a real Paris claim",
                           any_claim(lambda c: "paris" in (c["object_text"] or "").lower()
                                     or "paris" in (c["subject_text"] or "").lower())))

    elif arc == "derived_aggregate":
        checks.append(("frequency assertion stored as derived",
                       any_claim(lambda c: c["derived"] == 1)))

    passed = all(ok for _, ok in checks)
    return {"passed": passed, "checks": checks, "n_claims": len(claims)}
