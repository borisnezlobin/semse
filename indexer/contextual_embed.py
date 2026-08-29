"""contextual_embed.py — Phase 4: fold consolidation contexts into retrieval.

The consolidation scan writes 50-100-token situational headers into
`chunk_contexts` (memory.db), keyed by a durable (source, row-span) so they
survive full rebuilds. This module re-links each context to the chunk that now
holds the majority of its rows, writes that context onto `chunks.context`
(metadata.db), and folds it into both retrieval arms:

  dense  — re-embed `context + "\\n\\n" + chunk_text` (plain text where no
           context) and rebuild index.faiss + id_map.json so anaphoric chunks
           ("it's at 7") become retrievable by their referent.
  bm25   — rebuild chunks_fts over `context || ' ' || text`.

Display text is untouched: _hydrate reads chunks.text / chunks.messages and the
quote validator's haystack is built from those same columns — never from
chunks.context or chunks_fts. Contextualization only changes what *retrieves* a
chunk, never what is shown or quoted.

Re-link rule (generic, boundary-shift tolerant): chunk boundaries change every
full rebuild, so a context is assigned to the current chunk that contains the
MAJORITY of the rows in its span. The count is restricted to the context's own
thread — global chat.db rowids interleave threads, so a small thread's context
can span a huge rowid range full of other threads' rows; only same-thread rows
vote. Ties break on chunk_id ascending (deterministic). A context whose rows now
straddle several chunks lands on the one holding the plurality; several contexts
landing on one chunk are concatenated in row order.

`--new-only` (nightly): a per-chunk context_hash over the assembled context text
lets us re-embed only chunks whose context actually changed, replacing their
vectors in place instead of rebuilding every embedding.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
METADATA_DB_PATH = DATA_DIR / "metadata.db"
MEMORY_DB_PATH = DATA_DIR / "memory.db"
INDEX_PATH = DATA_DIR / "index.faiss"
ID_MAP_PATH = DATA_DIR / "id_map.json"

# MiniLM-L6 on batch=256 sustains this many texts/sec on Apple Silicon (MPS).
# Used only for the dry-run wall-clock estimate; the real pass measures itself.
_EMBED_THROUGHPUT = 800.0


# --------------------------------------------------------------------------- #
# Re-link: chunk_contexts (memory.db) -> current chunks (metadata.db)          #
# --------------------------------------------------------------------------- #

@dataclass
class _Ctx:
    context_id: str
    row_start: int
    row_end: int
    text: str


@dataclass
class RelinkStats:
    total: int = 0
    relinked: int = 0
    orphaned: int = 0          # no same-thread rows fell in the span
    ambiguous: int = 0         # top-two chunks tied on row count (broke by chunk_id)
    multi_context_chunks: int = 0


def _open(metadata_db: Path, memory_db: Path) -> sqlite3.Connection:
    """metadata.db writable; memory.db attached read-only for the row map."""
    conn = sqlite3.connect(metadata_db)
    conn.row_factory = sqlite3.Row
    conn.execute("ATTACH DATABASE ? AS mem",
                 (f"file:{memory_db}?mode=ro",))
    return conn


def majority_chunk(conn: sqlite3.Connection, source: str, row_start: int,
                   row_end: int, thread_key: str | None) -> str | None:
    """Current chunk holding the most of this span's same-thread rows, or None.

    Restricting to `thread_key` is what makes this correct despite interleaved
    global rowids: only rows that currently live in a chunk of the SAME thread
    are counted, so other threads' messages sharing the rowid range never win.
    """
    rows = conn.execute(
        "SELECT m.chunk_id AS cid, COUNT(*) AS n "
        "FROM mem.rowid_chunk_map m JOIN chunks c ON c.chunk_id = m.chunk_id "
        "WHERE m.source = ? AND m.row_id BETWEEN ? AND ? AND c.thread_key IS ? "
        "GROUP BY m.chunk_id ORDER BY n DESC, m.chunk_id ASC",
        (source, row_start, row_end, thread_key),
    ).fetchall()
    if not rows:
        return None
    return rows[0]["cid"]


def _is_ambiguous(conn: sqlite3.Connection, source: str, row_start: int,
                  row_end: int, thread_key: str | None) -> bool:
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM mem.rowid_chunk_map m "
        "JOIN chunks c ON c.chunk_id = m.chunk_id "
        "WHERE m.source = ? AND m.row_id BETWEEN ? AND ? AND c.thread_key IS ? "
        "GROUP BY m.chunk_id ORDER BY n DESC LIMIT 2",
        (source, row_start, row_end, thread_key),
    ).fetchall()
    return len(rows) > 1 and rows[0]["n"] == rows[1]["n"]


def relink(conn: sqlite3.Connection) -> tuple[dict[str, list[_Ctx]], RelinkStats]:
    """Assign every chunk_context to its majority current chunk.

    Returns (chunk_id -> contexts landing on it, stats). Contexts under a chunk
    are returned in row order so the assembled header reads chronologically.
    """
    stats = RelinkStats()
    mapping: dict[str, list[_Ctx]] = {}
    for cc in conn.execute(
        "SELECT context_id, source, row_start, row_end, thread_key, context "
        "FROM mem.chunk_contexts"
    ):
        stats.total += 1
        rs, re_ = cc["row_start"], cc["row_end"]
        if rs is None or re_ is None or not (cc["context"] or "").strip():
            stats.orphaned += 1
            continue
        if rs > re_:
            rs, re_ = re_, rs
        cid = majority_chunk(conn, cc["source"], rs, re_, cc["thread_key"])
        if cid is None:
            stats.orphaned += 1
            continue
        stats.relinked += 1
        if _is_ambiguous(conn, cc["source"], rs, re_, cc["thread_key"]):
            stats.ambiguous += 1
        mapping.setdefault(cid, []).append(
            _Ctx(cc["context_id"], rs, re_, cc["context"].strip()))
    for ctxs in mapping.values():
        ctxs.sort(key=lambda c: (c.row_start, c.row_end, c.context_id))
        if len(ctxs) > 1:
            stats.multi_context_chunks += 1
    return mapping, stats


def assemble_context(contexts: list[_Ctx]) -> str:
    """One header per chunk: its contexts joined in row order (deduped)."""
    seen: set[str] = set()
    parts: list[str] = []
    for c in contexts:
        if c.text not in seen:
            seen.add(c.text)
            parts.append(c.text)
    return " ".join(parts)


def context_hash(assembled: str) -> str:
    """Hash the exact embedded context text — identical text ⇒ skip re-embed."""
    return hashlib.sha1((assembled or "").encode("utf-8")).hexdigest()[:16]


def embed_input(context: str | None, text: str) -> str:
    """The string that gets embedded / FTS-indexed for a chunk."""
    if context and context.strip():
        return f"{context}\n\n{text}"
    return text


# --------------------------------------------------------------------------- #
# Change planning                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Plan:
    writes: list[tuple[str, str | None, str]] = field(default_factory=list)  # (chunk_id, context, hash)
    changed_ids: list[str] = field(default_factory=list)   # need re-embed
    cleared_ids: list[str] = field(default_factory=list)   # lost their context
    total_chunks: int = 0
    with_context: int = 0


def ensure_context_hash_column(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    if "context_hash" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN context_hash TEXT")
        conn.commit()


def plan_changes(conn: sqlite3.Connection, mapping: dict[str, list[_Ctx]],
                 new_only: bool) -> Plan:
    """Decide which chunks get a (re)written context and which need re-embedding.

    Full mode marks every chunk changed (rebuild everything). --new-only marks
    only chunks whose assembled context hash differs from what is stored, plus
    chunks that lost a context they previously had.
    """
    plan = Plan()
    stored: dict[str, tuple[str | None, str | None]] = {}
    for r in conn.execute("SELECT chunk_id, context, context_hash FROM chunks"):
        stored[r["chunk_id"]] = (r["context"], r["context_hash"])
    plan.total_chunks = len(stored)

    desired: dict[str, str] = {
        cid: assemble_context(ctxs) for cid, ctxs in mapping.items()}
    plan.with_context = sum(1 for v in desired.values() if v)

    for cid, (cur_ctx, cur_hash) in stored.items():
        want = desired.get(cid, "")
        want_hash = context_hash(want) if want else None
        if want:
            differs = cur_hash != want_hash
            plan.writes.append((cid, want, want_hash))
            if not new_only or differs:
                plan.changed_ids.append(cid)
        elif cur_ctx is not None or cur_hash is not None:
            # Previously carried a context; now maps to none — clear + re-embed.
            plan.writes.append((cid, None, None))
            plan.cleared_ids.append(cid)
            plan.changed_ids.append(cid)
        elif not new_only:
            # Plain chunk in full mode: no context row to write, but it is part
            # of the wholesale re-embed.
            plan.changed_ids.append(cid)
    return plan


# --------------------------------------------------------------------------- #
# Writers: chunks.context, chunks_fts                                          #
# --------------------------------------------------------------------------- #

def write_contexts(conn: sqlite3.Connection, plan: Plan) -> None:
    conn.executemany(
        "UPDATE chunks SET context = ?, context_hash = ? WHERE chunk_id = ?",
        [(ctx, h, cid) for (cid, ctx, h) in plan.writes],
    )
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection) -> int:
    """Rebuild chunks_fts over contextualized text.

    Wholesale DELETE + INSERT..SELECT. `COALESCE(context || ' ', '')` yields ''
    for chunks without a context (NULL propagates through ||), so plain chunks
    index exactly their text as before. Cheap (~65k rows) and independent of the
    embedding pass, so it runs in every mode.
    """
    conn.execute("DELETE FROM chunks_fts")
    conn.execute(
        "INSERT INTO chunks_fts (chunk_id, text) "
        "SELECT chunk_id, COALESCE(context || ' ', '') || text FROM chunks"
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]


# --------------------------------------------------------------------------- #
# Dense index rebuild                                                          #
# --------------------------------------------------------------------------- #

def _embed_inputs(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[str]:
    placeholders = ",".join("?" * len(chunk_ids))
    by_id = {
        r["chunk_id"]: embed_input(r["context"], r["text"] or "")
        for r in conn.execute(
            f"SELECT chunk_id, text, context FROM chunks "
            f"WHERE chunk_id IN ({placeholders})", chunk_ids)
    }
    return [by_id[cid] for cid in chunk_ids]


def rebuild_index_full(conn: sqlite3.Connection, embedder, index_path: Path,
                       id_map_path: Path) -> int:
    """Re-embed every chunk in a stable order and write a fresh FAISS index.

    Order (chunks.rowid) is arbitrary but internally consistent: id_map[i] names
    the chunk at FAISS position i, which is all the search layer requires.
    """
    import numpy as np
    import faiss
    from build_index import _build_faiss

    ordered = [r["chunk_id"] for r in
               conn.execute("SELECT chunk_id FROM chunks ORDER BY rowid")]
    inputs = _embed_inputs(conn, ordered)
    vectors = embedder.embed_batch(inputs).astype("float32")
    index = _build_faiss(vectors)
    faiss.write_index(index, str(index_path))
    id_map_path.write_text(json.dumps(ordered))
    return len(ordered)


def rebuild_index_incremental(conn: sqlite3.Connection, embedder,
                              changed_ids: list[str], index_path: Path,
                              id_map_path: Path) -> int:
    """Replace only changed chunks' vectors in the existing index.

    Reconstructs the current matrix, re-embeds the changed chunks, and rebuilds a
    fresh Flat index from the patched matrix — keeping id_map order (hence every
    unchanged embedding) intact. Chunks absent from id_map (e.g. appended by
    `build --update` but never given a context) are appended.
    """
    import numpy as np
    import faiss
    from build_index import _build_faiss

    index = faiss.read_index(str(index_path))
    id_map: list[str] = json.loads(id_map_path.read_text())
    pos = {cid: i for i, cid in enumerate(id_map)}
    if isinstance(index, faiss.IndexIVF):
        index.make_direct_map()  # reconstruct_n needs a direct map on IVF
    matrix = index.reconstruct_n(0, index.ntotal)

    in_index = [cid for cid in changed_ids if cid in pos]
    appended = [cid for cid in changed_ids if cid not in pos]

    if in_index:
        vecs = embedder.embed_batch(_embed_inputs(conn, in_index)).astype("float32")
        for cid, v in zip(in_index, vecs):
            matrix[pos[cid]] = v
    if appended:
        add_vecs = embedder.embed_batch(_embed_inputs(conn, appended)).astype("float32")
        matrix = np.vstack([matrix, add_vecs])
        id_map = id_map + appended

    new_index = _build_faiss(matrix)
    faiss.write_index(new_index, str(index_path))
    id_map_path.write_text(json.dumps(id_map))
    return len(in_index) + len(appended)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def run(*, metadata_db: Path = METADATA_DB_PATH, memory_db: Path = MEMORY_DB_PATH,
        index_path: Path = INDEX_PATH, id_map_path: Path = ID_MAP_PATH,
        new_only: bool = False, dry_run: bool = False) -> dict:
    """Contextualize the index. Returns a report dict.

    dry_run reports the re-link + change counts and a sizing estimate WITHOUT
    embedding, rebuilding FAISS, or writing chunks.context — the fast validation
    path and the memory-safe path while other work owns the model.
    """
    conn = _open(metadata_db, memory_db)
    try:
        ensure_context_hash_column(conn)
        mapping, stats = relink(conn)
        plan = plan_changes(conn, mapping, new_only)

        n_reembed = len(plan.changed_ids)
        report = {
            "contexts_total": stats.total,
            "contexts_relinked": stats.relinked,
            "contexts_orphaned": stats.orphaned,
            "contexts_ambiguous": stats.ambiguous,
            "chunks_with_context": plan.with_context,
            "chunks_multi_context": stats.multi_context_chunks,
            "chunks_total": plan.total_chunks,
            "chunks_to_reembed": n_reembed,
            "chunks_cleared": len(plan.cleared_ids),
            "mode": "new-only" if new_only else "full",
            "est_embed_seconds": round(n_reembed / _EMBED_THROUGHPUT, 1),
            "dry_run": dry_run,
        }
        if dry_run:
            return report

        write_contexts(conn, plan)
        report["fts_rows"] = rebuild_fts(conn)

        from embedder import Embedder
        embedder = Embedder()
        if new_only:
            report["vectors_replaced"] = rebuild_index_incremental(
                conn, embedder, plan.changed_ids, index_path, id_map_path)
        else:
            report["vectors_written"] = rebuild_index_full(
                conn, embedder, index_path, id_map_path)
        return report
    finally:
        conn.close()
