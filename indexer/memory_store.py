"""memory.db — the consolidation store that lives alongside the FAISS index.

metadata.db is wiped on every full rebuild, so all state that must survive a rebuild
lives here instead. Phase 0 seeded the `rowid_chunk_map`: a lookup from a durable
(source, row_id) pair to the chunk that currently contains that message. chunk_ids are
regenerated as fresh uuids each full rebuild, so any state keyed on row_ids reaches the
live chunk through this map, which is refreshed after every build.

Phase 2 adds the knowledge store the consolidation scan writes into: entities, aliases
(with validity windows + FTS for demand paging), claims (bi-temporal — invalidated,
never deleted), entity_mentions (the aggregation index), per-block scan checkpoints and
snapshots, chunk contexts (keyed by row span so they survive rebuilds), and telemetry.

Everything the scan keys on is (source, row_id) + dates, never chunk_id.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MEMORY_DB_PATH = DATA_DIR / "memory.db"

ENT_ME = "ent_me"  # the pinned identity for the corpus owner

_MAP_BATCH = 10_000


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rowid_chunk_map (
    source   TEXT NOT NULL,
    row_id   INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (source, row_id)
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id      TEXT PRIMARY KEY,
    kind           TEXT,               -- free-text: person | project | event | activity | ...
    canonical_name TEXT NOT NULL,
    summary        TEXT,               -- filled by Phase 7 dossiers; null at scan time
    contact_key    TEXT,               -- address-book key when the entity is a known contact
    first_seen     TEXT,
    last_seen      TEXT,
    merged_into    TEXT,               -- tombstone set by Phase 3 global merge
    created_at     TEXT,
    updated_at     TEXT
);

CREATE TABLE IF NOT EXISTS aliases (
    alias_id      TEXT PRIMARY KEY,
    entity_id     TEXT NOT NULL,
    surface       TEXT NOT NULL,       -- verbatim as it appeared
    norm          TEXT NOT NULL,       -- normalized for dedupe/lookup
    kind          TEXT NOT NULL,       -- name | nickname | description | handle
    speaker       TEXT,                -- entity_id whose viewpoint owns a description alias
    first_seen    TEXT,
    last_seen     TEXT,
    valid_from    TEXT,                -- alias validity window (drift: cadence -> Sendcraft)
    valid_until   TEXT,
    source_thread TEXT,
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_aliases_entity ON aliases(entity_id);
CREATE INDEX IF NOT EXISTS idx_aliases_norm   ON aliases(norm);

-- Demand-paging index: term -> entity. Kept in sync on every alias upsert.
CREATE VIRTUAL TABLE IF NOT EXISTS aliases_fts USING fts5(
    surface, norm, entity_id UNINDEXED, alias_kind UNINDEXED
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id TEXT NOT NULL,
    source    TEXT NOT NULL,
    row_id    INTEGER NOT NULL,        -- representative row of the chunk-situation
    date_iso  TEXT,
    PRIMARY KEY (entity_id, source, row_id)
);
CREATE INDEX IF NOT EXISTS idx_mentions_entity_date ON entity_mentions(entity_id, date_iso);

CREATE TABLE IF NOT EXISTS claims (
    claim_id       TEXT PRIMARY KEY,
    subject_id     TEXT,               -- resolved entity; null for compound/unresolved subjects
    subject_text   TEXT,               -- verbatim subject string ("Me and Sam") — always kept
    relation       TEXT,               -- free-text predicate
    object_text    TEXT,
    object_id      TEXT,               -- resolved when object_ref names a known entity
    asserted_by    TEXT,               -- speaker name as stated
    asserted_by_id TEXT,               -- resolved speaker entity (ent_me for the user)
    stated_at      TEXT,               -- when it was said (message timestamp)
    valid_from     TEXT,
    valid_until    TEXT,               -- bounded states expire here
    created_at     TEXT,
    invalidated_at TEXT,               -- set when superseded — never deleted
    superseded_by  TEXT,               -- claim_id that replaced this one
    confidence     REAL,
    derived        INTEGER DEFAULT 0,  -- aggregate/statistical assertion — computed value wins
    evidence       TEXT,               -- JSON [{source, row_id, quote}]
    thread_key     TEXT,
    scan_run_id    TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_id);
CREATE INDEX IF NOT EXISTS idx_claims_subjtext ON claims(subject_text);
CREATE INDEX IF NOT EXISTS idx_claims_valid ON claims(invalidated_at);

-- Undeclared same-subject collisions land here — never auto-invalidated.
CREATE TABLE IF NOT EXISTS claim_review_queue (
    id                TEXT PRIMARY KEY,
    thread_key        TEXT,
    subject_id        TEXT,
    subject_text      TEXT,
    relation          TEXT,
    existing_claim_id TEXT,
    new_claim_id      TEXT,
    reason            TEXT,
    created_at        TEXT,
    resolved          INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunk_contexts (
    context_id  TEXT PRIMARY KEY,
    thread_key  TEXT,
    source      TEXT,
    row_start   INTEGER,
    row_end     INTEGER,
    date_start  TEXT,
    date_end    TEXT,
    context     TEXT,
    entities    TEXT,                  -- JSON array of entity_ids referenced
    scan_run_id TEXT,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_contexts_span ON chunk_contexts(source, row_start, row_end);

CREATE TABLE IF NOT EXISTS scan_checkpoints (
    thread_key      TEXT PRIMARY KEY,
    watermark_iso   TEXT,              -- last consolidated message timestamp
    watermark_row_id INTEGER,          -- last consolidated row_id
    blocks_done     INTEGER DEFAULT 0,
    state_json      TEXT,              -- rolling working-set state to resume from
    status          TEXT,              -- running | done | error
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS thread_state_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    thread_key  TEXT,
    block_index INTEGER,
    state_json  TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS scan_log (
    id            TEXT PRIMARY KEY,
    scan_run_id   TEXT,
    thread_key    TEXT,
    block_index   INTEGER,
    model         TEXT,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    n_messages    INTEGER,
    n_entities    INTEGER,
    n_claims      INTEGER,
    n_mentions    INTEGER,
    evidence_dropped INTEGER,
    claims_dropped   INTEGER,
    review_queued    INTEGER,
    duration_s    REAL,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS merge_log (
    id          TEXT PRIMARY KEY,
    survivor_id TEXT,
    merged_id   TEXT,
    reason      TEXT,
    created_at  TEXT,
    reverted    INTEGER DEFAULT 0
);

-- Vision captions for image/video attachments. Keyed by the durable chat.db
-- attachment ROWID: a caption describes fixed pixels, so it is computed once and
-- reused across every scan, cell, and re-run. NULL caption records a permanent
-- failure (missing file, unreadable codec) so we never re-attempt it.
CREATE TABLE IF NOT EXISTS captions (
    attachment_id INTEGER,
    source        TEXT,
    path          TEXT,
    mime          TEXT,
    caption       TEXT,
    created_at    TEXT,
    PRIMARY KEY (source, attachment_id)
);
"""


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
    _ensure_ent_me(conn)


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open (creating if needed) memory.db with the full schema in place."""
    conn = sqlite3.connect(path or MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    _init(conn)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# --------------------------------------------------------------------------- #
# Normalization (script-agnostic so Cyrillic evidence is reachable)            #
# --------------------------------------------------------------------------- #

def norm_surface(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Keeps unicode word chars."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_for_match(s: str) -> str:
    """Gentle normalization for verbatim-quote verification (keeps word boundaries)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"')
          .replace("—", "-").replace("–", "-")
          .replace("￼", " "))  # ￼ attachment placeholder -> space, so caption-adjacent
    s = s.lower()                    #   quotes match whether or not the ￼ was stripped
    s = re.sub(r"\s+", " ", s).strip()
    return s


def verify_quote(quote: str, source_text: str) -> bool:
    """True when the evidence quote is a normalized substring of the source message.

    Script-agnostic: normalization only touches whitespace, case, and quote/dash
    glyph variants, so an original-script (e.g. Cyrillic) quote still validates.
    """
    q = _norm_for_match(quote)
    if not q:
        return False
    return q in _norm_for_match(source_text)


# --------------------------------------------------------------------------- #
# rowid_chunk_map (Phase 0 — regenerated after every build)                    #
# --------------------------------------------------------------------------- #

def refresh_rowid_chunk_map(
    metadata_db_path: Path | str,
    memory_db_path: Path | str | None = None,
) -> int:
    """Rebuild rowid_chunk_map from the current chunks table. Returns row count."""
    conn = connect(memory_db_path)
    try:
        conn.execute("DELETE FROM rowid_chunk_map")
        src = sqlite3.connect(f"file:{metadata_db_path}?mode=ro", uri=True)
        try:
            batch: list[tuple[str, int, str]] = []
            for chunk_id, source, row_ids_json in src.execute(
                "SELECT chunk_id, source, row_ids FROM chunks"
            ):
                for row_id in json.loads(row_ids_json or "[]"):
                    batch.append((source, row_id, chunk_id))
                    if len(batch) >= _MAP_BATCH:
                        _write_map_batch(conn, batch)
                        batch = []
            if batch:
                _write_map_batch(conn, batch)
            conn.commit()
        finally:
            src.close()
        return conn.execute("SELECT COUNT(*) FROM rowid_chunk_map").fetchone()[0]
    finally:
        conn.close()


def _write_map_batch(conn: sqlite3.Connection, batch: list[tuple[str, int, str]]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO rowid_chunk_map (source, row_id, chunk_id) VALUES (?, ?, ?)",
        batch,
    )


# --------------------------------------------------------------------------- #
# Entities                                                                     #
# --------------------------------------------------------------------------- #

def _ensure_ent_me(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT 1 FROM entities WHERE entity_id=?", (ENT_ME,)).fetchone():
        return
    now = _now()
    conn.execute(
        "INSERT INTO entities (entity_id, kind, canonical_name, created_at, updated_at) "
        "VALUES (?, 'person', 'Me', ?, ?)",
        (ENT_ME, now, now),
    )
    conn.commit()


def find_entity_by_norm(
    conn: sqlite3.Connection, norm: str, kind: str | None = None
) -> str | None:
    """Resolve a name to an existing entity by canonical or alias norm.

    Alias matches unify handle<->name identity ("jdoe" -> Jordan) coined in any block.
    A kind filter, when given, only narrows canonical matches; alias matches ignore it
    because an alias carries no independent kind.
    """
    if not norm:
        return None
    if kind:
        row = conn.execute(
            "SELECT entity_id FROM entities WHERE merged_into IS NULL "
            "AND lower(canonical_name)=? AND kind=? LIMIT 1",
            (norm, kind),
        ).fetchone()
        if row:
            return row[0]
    row = conn.execute(
        "SELECT e.entity_id FROM entities e WHERE e.merged_into IS NULL "
        "AND lower(e.canonical_name)=? LIMIT 1",
        (norm,),
    ).fetchone()
    if row:
        return row[0]
    row = conn.execute(
        "SELECT a.entity_id FROM aliases a JOIN entities e ON a.entity_id=e.entity_id "
        "WHERE e.merged_into IS NULL AND a.norm=? "
        "AND a.kind IN ('name','nickname','handle') LIMIT 1",
        (norm,),
    ).fetchone()
    return row[0] if row else None


def create_entity(
    conn: sqlite3.Connection,
    canonical_name: str,
    kind: str,
    contact_key: str | None = None,
    first_seen: str | None = None,
    last_seen: str | None = None,
) -> str:
    entity_id = _new_id()
    now = _now()
    conn.execute(
        "INSERT INTO entities (entity_id, kind, canonical_name, contact_key, "
        "first_seen, last_seen, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (entity_id, kind, canonical_name, contact_key, first_seen, last_seen, now, now),
    )
    return entity_id


def touch_entity_seen(conn: sqlite3.Connection, entity_id: str, date_iso: str | None) -> None:
    """Widen first_seen/last_seen to include date_iso."""
    if not date_iso:
        return
    row = conn.execute(
        "SELECT first_seen, last_seen FROM entities WHERE entity_id=?", (entity_id,)
    ).fetchone()
    if not row:
        return
    first, last = row[0], row[1]
    new_first = date_iso if (first is None or date_iso < first) else first
    new_last = date_iso if (last is None or date_iso > last) else last
    conn.execute(
        "UPDATE entities SET first_seen=?, last_seen=?, updated_at=? WHERE entity_id=?",
        (new_first, new_last, _now(), entity_id),
    )


def set_entity_contact_key(conn: sqlite3.Connection, entity_id: str, contact_key: str) -> None:
    conn.execute(
        "UPDATE entities SET contact_key=? WHERE entity_id=? AND contact_key IS NULL",
        (contact_key, entity_id),
    )


# --------------------------------------------------------------------------- #
# Aliases                                                                      #
# --------------------------------------------------------------------------- #

def upsert_alias(
    conn: sqlite3.Connection,
    entity_id: str,
    surface: str,
    kind: str,
    speaker: str | None = None,
    date_iso: str | None = None,
    valid_from: str | None = None,
    valid_until: str | None = None,
    source_thread: str | None = None,
) -> None:
    """Bind a surface form to an entity, deduping on (entity_id, norm, kind, speaker).

    Repeated sightings widen the first_seen/last_seen window rather than duplicating.
    """
    norm = norm_surface(surface)
    if not norm:
        return
    existing = conn.execute(
        "SELECT alias_id, first_seen, last_seen, valid_from, valid_until FROM aliases "
        "WHERE entity_id=? AND norm=? AND kind=? AND IFNULL(speaker,'')=IFNULL(?,'')",
        (entity_id, norm, kind, speaker),
    ).fetchone()
    if existing:
        alias_id, first, last, vf, vu = existing
        new_first = date_iso if (date_iso and (first is None or date_iso < first)) else first
        new_last = date_iso if (date_iso and (last is None or date_iso > last)) else last
        conn.execute(
            "UPDATE aliases SET first_seen=?, last_seen=?, valid_from=COALESCE(?, valid_from), "
            "valid_until=COALESCE(?, valid_until) WHERE alias_id=?",
            (new_first, new_last, valid_from, valid_until, alias_id),
        )
        return
    alias_id = _new_id()
    conn.execute(
        "INSERT INTO aliases (alias_id, entity_id, surface, norm, kind, speaker, "
        "first_seen, last_seen, valid_from, valid_until, source_thread, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (alias_id, entity_id, surface, norm, kind, speaker,
         date_iso, date_iso, valid_from, valid_until, source_thread, _now()),
    )
    conn.execute(
        "INSERT INTO aliases_fts (surface, norm, entity_id, alias_kind) VALUES (?, ?, ?, ?)",
        (surface, norm, entity_id, kind),
    )


# --------------------------------------------------------------------------- #
# Demand paging — surface known entities relevant to a block's candidate terms  #
# --------------------------------------------------------------------------- #

def _fts_or_query(terms: list[str]) -> str:
    quoted = []
    for t in terms:
        t = re.sub(r'"', " ", t).strip()
        if t:
            quoted.append(f'"{t}"')
    return " OR ".join(quoted)


def page_entities(
    conn: sqlite3.Connection,
    terms: list[str],
    pinned_entity_ids: list[str],
    limit: int = 12,
) -> list[dict]:
    """Return compact records for entities matching the block's candidate terms.

    Always includes the pinned (participant + ent_me) entities so the model can bind
    to the thread's people even when no candidate term matched them.
    """
    entity_ids: list[str] = list(dict.fromkeys(pinned_entity_ids))
    match = _fts_or_query(terms)
    if match:
        try:
            for (eid,) in conn.execute(
                "SELECT DISTINCT entity_id FROM aliases_fts WHERE aliases_fts MATCH ? LIMIT ?",
                (match, limit * 3),
            ):
                if eid not in entity_ids:
                    entity_ids.append(eid)
        except sqlite3.OperationalError:
            pass  # malformed FTS term set — pinned entities still returned
    entity_ids = entity_ids[: limit + len(pinned_entity_ids)]
    out: list[dict] = []
    for eid in entity_ids:
        ent = conn.execute(
            "SELECT entity_id, kind, canonical_name, first_seen, last_seen "
            "FROM entities WHERE entity_id=? AND merged_into IS NULL",
            (eid,),
        ).fetchone()
        if not ent:
            continue
        aliases = [
            {"surface": r[0], "kind": r[1]}
            for r in conn.execute(
                "SELECT surface, kind FROM aliases WHERE entity_id=? "
                "ORDER BY kind LIMIT 8",
                (eid,),
            )
        ]
        out.append({
            "entity_id": ent[0],
            "canonical_name": ent[2],
            "kind": ent[1],
            "aliases": aliases,
        })
    return out


# --------------------------------------------------------------------------- #
# Claims                                                                       #
# --------------------------------------------------------------------------- #

def insert_claim(
    conn: sqlite3.Connection,
    *,
    subject_id: str | None,
    subject_text: str | None,
    relation: str | None,
    object_text: str | None,
    object_id: str | None,
    asserted_by: str | None,
    asserted_by_id: str | None,
    stated_at: str | None,
    valid_from: str | None,
    valid_until: str | None,
    confidence: float | None,
    derived: bool,
    evidence: list[dict],
    thread_key: str | None,
    scan_run_id: str | None,
) -> str:
    claim_id = _new_id()
    conn.execute(
        "INSERT INTO claims (claim_id, subject_id, subject_text, relation, object_text, "
        "object_id, asserted_by, asserted_by_id, stated_at, valid_from, valid_until, "
        "created_at, confidence, derived, evidence, thread_key, scan_run_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (claim_id, subject_id, subject_text, relation, object_text, object_id,
         asserted_by, asserted_by_id, stated_at, valid_from, valid_until, _now(),
         confidence, 1 if derived else 0, json.dumps(evidence), thread_key, scan_run_id),
    )
    return claim_id


def find_supersedable_claim(
    conn: sqlite3.Connection,
    subject_id: str | None,
    subject_text: str | None,
    relation: str | None,
) -> sqlite3.Row | None:
    """Most-recent still-valid claim with the same subject+relation, if any."""
    rel = norm_surface(relation or "")
    if subject_id:
        row = conn.execute(
            "SELECT * FROM claims WHERE subject_id=? AND invalidated_at IS NULL "
            "AND lower(relation)=? ORDER BY stated_at DESC LIMIT 1",
            (subject_id, rel),
        ).fetchone()
        if row:
            return row
    if subject_text:
        return conn.execute(
            "SELECT * FROM claims WHERE subject_id IS NULL AND lower(subject_text)=? "
            "AND invalidated_at IS NULL AND lower(relation)=? "
            "ORDER BY stated_at DESC LIMIT 1",
            (subject_text.lower(), rel),
        ).fetchone()
    return None


def supersede_claim(
    conn: sqlite3.Connection,
    old_claim_id: str,
    new_claim_id: str,
    invalidated_at: str | None,
    valid_until: str | None,
) -> None:
    conn.execute(
        "UPDATE claims SET superseded_by=?, invalidated_at=?, "
        "valid_until=COALESCE(valid_until, ?) WHERE claim_id=?",
        (new_claim_id, invalidated_at, valid_until, old_claim_id),
    )


def add_to_review_queue(
    conn: sqlite3.Connection,
    thread_key: str | None,
    subject_id: str | None,
    subject_text: str | None,
    relation: str | None,
    existing_claim_id: str,
    new_claim_id: str,
    reason: str,
) -> None:
    conn.execute(
        "INSERT INTO claim_review_queue (id, thread_key, subject_id, subject_text, "
        "relation, existing_claim_id, new_claim_id, reason, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (_new_id(), thread_key, subject_id, subject_text, relation,
         existing_claim_id, new_claim_id, reason, _now()),
    )


# --------------------------------------------------------------------------- #
# Entity mentions / chunk contexts                                             #
# --------------------------------------------------------------------------- #

def insert_entity_mention(
    conn: sqlite3.Connection, entity_id: str, source: str, row_id: int, date_iso: str | None
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO entity_mentions (entity_id, source, row_id, date_iso) "
        "VALUES (?, ?, ?, ?)",
        (entity_id, source, row_id, date_iso),
    )


def insert_chunk_context(
    conn: sqlite3.Connection,
    *,
    thread_key: str,
    source: str,
    row_start: int,
    row_end: int,
    date_start: str | None,
    date_end: str | None,
    context: str,
    entity_ids: list[str],
    scan_run_id: str | None,
) -> None:
    conn.execute(
        "INSERT INTO chunk_contexts (context_id, thread_key, source, row_start, row_end, "
        "date_start, date_end, context, entities, scan_run_id, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (_new_id(), thread_key, source, row_start, row_end, date_start, date_end,
         context, json.dumps(entity_ids), scan_run_id, _now()),
    )


# --------------------------------------------------------------------------- #
# Checkpoints / snapshots / telemetry                                          #
# --------------------------------------------------------------------------- #

def get_checkpoint(conn: sqlite3.Connection, thread_key: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM scan_checkpoints WHERE thread_key=?", (thread_key,)
    ).fetchone()


def upsert_checkpoint(
    conn: sqlite3.Connection,
    thread_key: str,
    watermark_iso: str | None,
    watermark_row_id: int | None,
    blocks_done: int,
    state_json: str,
    status: str,
) -> None:
    conn.execute(
        "INSERT INTO scan_checkpoints (thread_key, watermark_iso, watermark_row_id, "
        "blocks_done, state_json, status, updated_at) VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(thread_key) DO UPDATE SET watermark_iso=excluded.watermark_iso, "
        "watermark_row_id=excluded.watermark_row_id, blocks_done=excluded.blocks_done, "
        "state_json=excluded.state_json, status=excluded.status, updated_at=excluded.updated_at",
        (thread_key, watermark_iso, watermark_row_id, blocks_done, state_json, status, _now()),
    )


def insert_snapshot(
    conn: sqlite3.Connection, thread_key: str, block_index: int, state_json: str
) -> None:
    conn.execute(
        "INSERT INTO thread_state_snapshots (snapshot_id, thread_key, block_index, "
        "state_json, created_at) VALUES (?,?,?,?,?)",
        (_new_id(), thread_key, block_index, state_json, _now()),
    )


def insert_scan_log(conn: sqlite3.Connection, **fields) -> None:
    fields.setdefault("id", _new_id())
    fields.setdefault("created_at", _now())
    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" * len(fields))
    conn.execute(
        f"INSERT INTO scan_log ({cols}) VALUES ({placeholders})", tuple(fields.values())
    )


# --------------------------------------------------------------------------- #
# Attachment captions (vision) — computed once, reused forever                 #
# --------------------------------------------------------------------------- #

# Sentinel returned when an attachment is cached as a permanent failure. Callers
# distinguish "no cache row yet" (None) from "cached, but uncaptionable" (this).
CAPTION_FAILED = "\x00__caption_failed__"


def get_caption(conn: sqlite3.Connection, source: str, attachment_id: int) -> str | None:
    """Cached caption for an attachment, CAPTION_FAILED if a prior attempt failed,
    or None if it has never been attempted."""
    row = conn.execute(
        "SELECT caption FROM captions WHERE source=? AND attachment_id=?",
        (source, attachment_id),
    ).fetchone()
    if row is None:
        return None
    return row[0] if row[0] is not None else CAPTION_FAILED


def put_caption(
    conn: sqlite3.Connection,
    source: str,
    attachment_id: int,
    path: str | None,
    mime: str | None,
    caption: str | None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO captions (attachment_id, source, path, mime, caption, "
        "created_at) VALUES (?,?,?,?,?,?)",
        (attachment_id, source, path, mime, caption, _now()),
    )
