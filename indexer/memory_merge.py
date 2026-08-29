"""Global entity merge — cross-thread consolidation of the knowledge store.

The consolidation scan runs thread-by-thread, so the same real-world entity is
minted afresh in every thread that mentions it: "Smaran" in one thread and the
unknown handle "Unknown (•0725)" in another are two entity rows for one person;
a project renamed across an era drifts into two nodes. This pass finds those
duplicates and folds them into one, without ever destroying evidence.

Three cheap signals nominate candidate pairs (no LLM):
  - alias-norm collisions: two live entities of the same kind share a name/
    nickname surface after normalization.
  - MiniLM name neighbors: "canonical_name (kind) summary" embeddings within
    0.80 cosine, same-kind only. The ~200 entity NAMES are embedded once.
  - contact_key equality: the same address-book person — auto-merged, no
    adjudication (the address book is ground truth).

gpt-4o-mini adjudicates the nominated pairs in small batches. Its hardest job is
the *sibling trap*: same owner + same kind + similar descriptions are often
DISTINCT coexisting things (one person's two VS Code extensions; two separate
projects). The prompt looks for coexistence evidence and defaults such pairs to
`distinct`. Only a `merge` verdict merges; `unsure` lands in a review queue.

Every merge is one transaction, fully logged with a pre-merge JSON payload, so
`unmerge()` restores the prior state exactly. Aliases, claims, mentions and
chunk-context entity lists are repointed loser -> survivor; the survivor is the
older entity (tie: more claims; ent_me always wins). The merged loser is
tombstoned via `merged_into` and skipped everywhere the live graph is read.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import me_identity
import memory_store as store

ADJUDICATION_MODEL = "gpt-4o-mini"
EMBED_THRESHOLD = 0.80
ADJUDICATION_BATCH = 20
CLAIM_SAMPLES = 5

# Opaque handle placeholders ("Unknown (•0725)") are phone-number identifiers, not
# descriptive names: MiniLM rates two DIFFERENT unknown numbers ~0.95 alike because
# only four digits differ against a shared template. Their real merge signal is an
# alias collision (a real name bound to the handle) or contact_key — never name
# similarity — so they are excluded from the embedding nominator.
_OPAQUE_NAME_RE = re.compile(r"^\s*unknown\b", re.IGNORECASE)


def _is_opaque_name(name: str | None) -> bool:
    return bool(name) and _OPAQUE_NAME_RE.match(name) is not None


# --------------------------------------------------------------------------- #
# Schema — additive, owned by this module (memory_store.py stays untouched)    #
# --------------------------------------------------------------------------- #

def ensure_schema(conn: sqlite3.Connection) -> None:
    """Add the merge bookkeeping this phase needs, idempotently.

    `merge_log.payload` carries the reversible pre-merge snapshot; `merge_runs`
    records incremental cutoffs; `merge_review` is the `unsure` adjudication queue.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(merge_log)")}
    if "payload" not in cols:
        conn.execute("ALTER TABLE merge_log ADD COLUMN payload TEXT")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS merge_runs (
            run_id       TEXT PRIMARY KEY,
            started_at   TEXT,
            finished_at  TEXT,
            cutoff_iso   TEXT,
            n_candidates INTEGER,
            n_auto       INTEGER,
            n_merged     INTEGER,
            n_unsure     INTEGER,
            n_distinct   INTEGER,
            applied      INTEGER
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS merge_review (
            id         TEXT PRIMARY KEY,
            run_id     TEXT,
            entity_a   TEXT,
            entity_b   TEXT,
            source     TEXT,
            score      REAL,
            reason     TEXT,
            created_at TEXT,
            resolved   INTEGER DEFAULT 0
        )"""
    )
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Candidate generation                                                         #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    a: str            # entity_id (canonically the smaller id)
    b: str
    source: str       # "alias" | "embedding" | "contact_key" | "identity"
    score: float = 0.0
    evidence: tuple = ()  # user-identity evidence, only for the ent_me track

    @staticmethod
    def of(x: str, y: str, source: str, score: float = 0.0) -> "Candidate":
        lo, hi = (x, y) if x <= y else (y, x)
        return Candidate(lo, hi, source, score)


# --------------------------------------------------------------------------- #
# User identity — gate for absorbing a user-name entity into ent_me            #
# --------------------------------------------------------------------------- #

_USER_IDENTITY: tuple[set, set] | None = None


def _user_identity() -> tuple[set, set]:
    """(emails, phones) belonging to the corpus owner, from me_identity. Cached;
    a discovery failure degrades to empty sets (nothing absorbs into ent_me)."""
    global _USER_IDENTITY
    if _USER_IDENTITY is None:
        try:
            _USER_IDENTITY = me_identity.discover()
        except Exception:  # noqa: BLE001 — AddressBook may be unavailable
            _USER_IDENTITY = (set(), set())
    return _USER_IDENTITY


def user_identity_evidence(
    conn: sqlite3.Connection, entity_id: str, user_emails: set, user_phones: set
) -> list[str]:
    """Evidence that `entity_id` IS the corpus owner: any of its alias or canonical
    surfaces that resolves to a known user email/phone. Empty => not the user, so
    it must NOT fold into ent_me (a same-surname cousin stays distinct)."""
    if entity_id == store.ENT_ME or (not user_emails and not user_phones):
        return []
    surfaces: list[tuple[str, str]] = [
        (r[0], r[1]) for r in conn.execute(
            "SELECT DISTINCT surface, kind FROM aliases WHERE entity_id=?", (entity_id,))
    ]
    ent = conn.execute(
        "SELECT canonical_name FROM entities WHERE entity_id=?", (entity_id,)).fetchone()
    if ent and ent[0]:
        surfaces.append((ent[0], "canonical_name"))
    evidence: list[str] = []
    seen: set[str] = set()
    for surface, kind in surfaces:
        if surface and me_identity.is_user_handle(surface, user_emails, user_phones):
            tag = f"{kind} {surface!r} is a known user email/phone"
            if tag not in seen:
                seen.add(tag)
                evidence.append(tag)
    return evidence


def _live_entities(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT entity_id, kind, canonical_name, summary, contact_key, first_seen, "
        "last_seen, created_at, updated_at FROM entities WHERE merged_into IS NULL"
    ).fetchall()


def _touched_since(conn: sqlite3.Connection, cutoff: str | None) -> set[str] | None:
    """Entity ids created or updated since `cutoff` (None => all, no filter)."""
    if not cutoff:
        return None
    rows = conn.execute(
        "SELECT entity_id FROM entities WHERE merged_into IS NULL "
        "AND (created_at >= ? OR updated_at >= ?)",
        (cutoff, cutoff),
    )
    return {r[0] for r in rows}


def alias_collision_candidates(conn: sqlite3.Connection) -> list[Candidate]:
    """Same normalized name/nickname surface on two same-kind live entities."""
    rows = conn.execute(
        "SELECT a.norm, a.entity_id, e.kind FROM aliases a "
        "JOIN entities e ON a.entity_id = e.entity_id "
        "WHERE e.merged_into IS NULL AND a.kind IN ('name', 'nickname')"
    ).fetchall()
    by_norm: dict[str, dict[str, str]] = {}
    for r in rows:
        by_norm.setdefault(r["norm"], {})[r["entity_id"]] = r["kind"]
    out: list[Candidate] = []
    for norm, members in by_norm.items():
        ids = list(members)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if members[ids[i]] == members[ids[j]]:  # same entity-kind
                    out.append(Candidate.of(ids[i], ids[j], "alias"))
    return out


def embedding_candidates(
    conn: sqlite3.Connection, embedder, threshold: float = EMBED_THRESHOLD
) -> list[Candidate]:
    """Same-kind entity pairs whose name embeddings are within `threshold` cosine.

    Embeds the ~200 entity NAMES once (never chunks). Vectors are normalized by
    the embedder, so cosine is a plain dot product.
    """
    ents = [e for e in _live_entities(conn) if not _is_opaque_name(e["canonical_name"])]
    if len(ents) < 2:
        return []
    texts = [_embed_text(e) for e in ents]
    vecs = embedder.embed_batch(texts)  # single batched pass
    import numpy as np

    out: list[Candidate] = []
    by_kind: dict[str, list[int]] = {}
    for idx, e in enumerate(ents):
        by_kind.setdefault(e["kind"] or "", []).append(idx)
    for kind, idxs in by_kind.items():
        if len(idxs) < 2:
            continue
        sub = np.asarray([vecs[i] for i in idxs], dtype="float32")
        sims = sub @ sub.T
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                score = float(sims[a][b])
                if score >= threshold:
                    out.append(Candidate.of(
                        ents[idxs[a]]["entity_id"], ents[idxs[b]]["entity_id"],
                        "embedding", round(score, 4)))
    return out


def _embed_text(e: sqlite3.Row) -> str:
    base = f"{e['canonical_name']} ({e['kind']})"
    return f"{base} {e['summary']}" if e["summary"] else base


def contact_key_candidates(conn: sqlite3.Connection) -> list[Candidate]:
    """Live entities sharing a contact_key — same address-book person, auto-merge."""
    rows = conn.execute(
        "SELECT contact_key, entity_id FROM entities "
        "WHERE merged_into IS NULL AND contact_key IS NOT NULL AND contact_key != '' "
        "ORDER BY contact_key"
    ).fetchall()
    by_key: dict[str, list[str]] = {}
    for r in rows:
        by_key.setdefault(r["contact_key"], []).append(r["entity_id"])
    out: list[Candidate] = []
    for _, ids in by_key.items():
        for i in range(1, len(ids)):
            out.append(Candidate.of(ids[0], ids[i], "contact_key", 1.0))
    return out


@dataclass
class CandidateSet:
    auto: list[Candidate] = field(default_factory=list)          # contact_key
    identity: list[Candidate] = field(default_factory=list)      # ent_me absorption
    adjudicate: list[Candidate] = field(default_factory=list)    # alias + embedding


def _other_side(c: Candidate) -> str | None:
    """The non-ent_me id of a pair that involves ent_me, else None."""
    if c.a == store.ENT_ME:
        return c.b
    if c.b == store.ENT_ME:
        return c.a
    return None


def gather_candidates(
    conn: sqlite3.Connection, embedder=None, cutoff: str | None = None,
    threshold: float = EMBED_THRESHOLD, user_emails: set | None = None,
    user_phones: set | None = None,
) -> CandidateSet:
    """All candidate pairs, deduped. Incremental cutoff keeps only pairs touching
    an entity created/updated since the last run (either side qualifies).

    Pairs involving ent_me are routed to the identity track and gated on
    user-identity evidence: they absorb into ent_me only when the other side
    carries a known user email/phone, and default to distinct otherwise.
    """
    user_emails = user_emails or set()
    user_phones = user_phones or set()
    touched = _touched_since(conn, cutoff)

    def keep(c: Candidate) -> bool:
        return touched is None or c.a in touched or c.b in touched

    auto: dict[frozenset, Candidate] = {}
    for c in contact_key_candidates(conn):
        if keep(c):
            auto[frozenset((c.a, c.b))] = c

    adjud: dict[frozenset, Candidate] = {}
    for c in alias_collision_candidates(conn):
        if keep(c):
            adjud[frozenset((c.a, c.b))] = c
    if embedder is not None:
        for c in embedding_candidates(conn, embedder, threshold):
            key = frozenset((c.a, c.b))
            if keep(c) and key not in adjud:  # alias collision is the stronger signal
                adjud[key] = c

    # Peel ent_me-involving pairs out of the LLM track into the evidence-gated one.
    identity: dict[frozenset, Candidate] = {}
    for key in list(adjud):
        other = _other_side(adjud[key])
        if other is not None:
            ev = user_identity_evidence(conn, other, user_emails, user_phones)
            identity[key] = replace(adjud[key], source="identity", evidence=tuple(ev))
            del adjud[key]

    # Proactively pair any entity carrying user identity with ent_me, even without
    # a shared alias — a mail identity's email handle need not collide with "Me".
    for (other,) in conn.execute(
        "SELECT entity_id FROM entities WHERE merged_into IS NULL AND entity_id != ?",
        (store.ENT_ME,)):
        key = frozenset((store.ENT_ME, other))
        if key in identity:
            continue
        if touched is not None and other not in touched and store.ENT_ME not in touched:
            continue
        ev = user_identity_evidence(conn, other, user_emails, user_phones)
        if ev:
            identity[key] = replace(
                Candidate.of(store.ENT_ME, other, "identity"), evidence=tuple(ev))

    for key in list(adjud):  # a pair auto-merged on contact_key needs no adjudication
        if key in auto:
            del adjud[key]
    return CandidateSet(auto=list(auto.values()), identity=list(identity.values()),
                        adjudicate=list(adjud.values()))


# --------------------------------------------------------------------------- #
# Mini-dossiers (one per candidate side) for adjudication                      #
# --------------------------------------------------------------------------- #

def mini_dossier(conn: sqlite3.Connection, entity_id: str) -> dict:
    ent = conn.execute(
        "SELECT canonical_name, kind, contact_key, first_seen, last_seen "
        "FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()
    aliases = [
        f"{r['surface']} ({r['kind']})"
        for r in conn.execute(
            "SELECT DISTINCT surface, kind FROM aliases WHERE entity_id = ? "
            "ORDER BY kind LIMIT 10", (entity_id,))
    ]
    claims = []
    for c in conn.execute(
        "SELECT relation, object_text, stated_at, thread_key, invalidated_at, valid_until "
        "FROM claims WHERE subject_id = ? ORDER BY (invalidated_at IS NOT NULL), "
        "stated_at DESC LIMIT ?", (entity_id, CLAIM_SAMPLES)):
        date = (c["stated_at"] or "")[:10]
        state = " [superseded]" if c["invalidated_at"] else ""
        claims.append(f"{date} {c['relation']}: {c['object_text']}{state}")
    threads = [r[0] for r in conn.execute(
        "SELECT DISTINCT thread_key FROM claims WHERE subject_id = ? "
        "AND thread_key IS NOT NULL LIMIT 6", (entity_id,))]
    return {
        "entity_id": entity_id,
        "name": ent["canonical_name"] if ent else entity_id,
        "kind": ent["kind"] if ent else "",
        "contact_key": ent["contact_key"] if ent else None,
        "seen": f"{(ent['first_seen'] or '?')[:10]}..{(ent['last_seen'] or '?')[:10]}" if ent else "",
        "aliases": aliases,
        "claims": claims,
        "threads": threads,
    }


def _dossier_text(d: dict, label: str) -> str:
    lines = [f"{label}. {d['name']}  [kind={d['kind']}]  seen {d['seen']}"]
    if d["aliases"]:
        lines.append("   aliases: " + ", ".join(d["aliases"]))
    if d["claims"]:
        lines.append("   claims:")
        lines += [f"     - {c}" for c in d["claims"]]
    else:
        lines.append("   claims: (none)")
    lines.append(f"   seen in {len(d['threads'])} thread(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM adjudication                                                             #
# --------------------------------------------------------------------------- #

ADJUDICATION_SYSTEM = """You decide whether two extracted entities are the SAME \
real-world thing and should be merged into one, or are DISTINCT.

You will see numbered PAIRS. Each pair has side A and side B with a name, kind, \
aliases, a few dated claims, and how many conversation threads it appears in.

Return one verdict per pair:
- "merge": A and B are unmistakably the same entity — one person under two labels \
(a name and an unknown phone handle; a nickname and a full name), the same project \
under a renamed title, the same place/org written two ways. The claims and dates \
must be consistent with a single thing.
- "distinct": different real-world things.
- "unsure": genuinely ambiguous — not enough evidence either way.

CRITICAL — the sibling trap. Same owner + same kind + similar wording usually means \
two SEPARATE coexisting creations, NOT a duplicate. If A and B are both projects, \
extensions, events, or documents made or attended by the same person, assume they \
are DISTINCT siblings unless the evidence shows they are literally the same artifact \
(same identifiers, one renamed into the other, one explicitly said to BE the other). \
Two VS Code extensions by one author, two class events on one day, two essays — \
these are distinct. Default such pairs to "distinct".

Prefer "distinct" or "unsure" over a wrong "merge": a bad merge collapses two real \
things. Only answer "merge" when identity is clear.

Reply with JSON only: {"results": [{"pair_id": <int>, "verdict": "...", "reason": "<short>"}]}."""

_ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["merge", "distinct", "unsure"]},
                    "reason": {"type": "string"},
                },
                "required": ["pair_id", "verdict", "reason"],
            },
        }
    },
    "required": ["results"],
}


@dataclass
class Verdict:
    candidate: Candidate
    verdict: str      # merge | distinct | unsure | unadjudicated
    reason: str


def _openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key)


def adjudicate(
    conn: sqlite3.Connection, candidates: list[Candidate], client=None,
    batch_size: int = ADJUDICATION_BATCH,
) -> list[Verdict]:
    """Batch candidates through gpt-4o-mini. Without a key every pair is 'unsure'
    (queued, never silently merged)."""
    if not candidates:
        return []
    client = client or _openai_client()
    if client is None:
        print("memory_merge: OPENAI_API_KEY not set — pairs left unadjudicated",
              file=sys.stderr)
        return [Verdict(c, "unsure", "no adjudication backend") for c in candidates]

    verdicts: list[Verdict] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        blocks = []
        for i, c in enumerate(batch):
            da = mini_dossier(conn, c.a)
            db = mini_dossier(conn, c.b)
            hint = f"(nominated by {c.source}" + (f", name-similarity {c.score}" if c.source == "embedding" else "") + ")"
            blocks.append(f"PAIR {i} {hint}\n{_dossier_text(da, 'A')}\n{_dossier_text(db, 'B')}")
        user = "\n\n".join(blocks)
        try:
            resp = client.chat.completions.create(
                model=ADJUDICATION_MODEL,
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "merge_verdicts",
                                    "schema": _ADJUDICATION_SCHEMA, "strict": False},
                },
                messages=[
                    {"role": "system", "content": ADJUDICATION_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            by_id = {int(r["pair_id"]): r for r in data.get("results", [])}
        except Exception as e:  # noqa: BLE001 — a failed batch degrades to unsure
            print(f"memory_merge: adjudication batch failed ({e}) — pairs -> unsure",
                  file=sys.stderr)
            by_id = {}
        for i, c in enumerate(batch):
            r = by_id.get(i)
            if r and r.get("verdict") in ("merge", "distinct", "unsure"):
                verdicts.append(Verdict(c, r["verdict"], (r.get("reason") or "").strip()))
            else:
                verdicts.append(Verdict(c, "unsure", "missing/invalid verdict"))
    return verdicts


# --------------------------------------------------------------------------- #
# Merge operation (one transaction, reversible)                               #
# --------------------------------------------------------------------------- #

def _claim_count(conn: sqlite3.Connection, entity_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM claims WHERE subject_id = ?", (entity_id,)).fetchone()[0]


def choose_survivor(conn: sqlite3.Connection, x: str, y: str) -> tuple[str, str]:
    """Return (survivor, loser). Older first_seen wins; tie -> more claims. ent_me
    always survives (the pinned corpus owner is never tombstoned)."""
    if x == store.ENT_ME:
        return x, y
    if y == store.ENT_ME:
        return y, x
    rows = {r["entity_id"]: r for r in conn.execute(
        "SELECT entity_id, first_seen, created_at FROM entities WHERE entity_id IN (?, ?)",
        (x, y))}
    fx = (rows[x]["first_seen"] or "9999")
    fy = (rows[y]["first_seen"] or "9999")
    if fx != fy:
        return (x, y) if fx < fy else (y, x)
    cx, cy = _claim_count(conn, x), _claim_count(conn, y)
    if cx != cy:
        return (x, y) if cx > cy else (y, x)
    # Stable final tie-break on creation order.
    return (x, y) if (rows[x]["created_at"] or "") <= (rows[y]["created_at"] or "") else (y, x)


def merge_entities(
    conn: sqlite3.Connection, survivor: str, loser: str, reason: str,
    run_id: str | None = None,
) -> str:
    """Fold `loser` into `survivor` in one transaction. Returns the merge_log id.

    Repoints every entity reference (aliases, claims, mentions, chunk contexts),
    tombstones the loser, re-runs the same-subject contradiction check over the
    union, and records a reversible pre-merge payload.
    """
    if survivor == loser:
        raise ValueError("cannot merge an entity into itself")
    ensure_schema(conn)

    payload: dict = {"survivor": survivor, "loser": loser}

    # -- capture what we are about to repoint (for exact unmerge) --
    payload["alias_ids"] = [r[0] for r in conn.execute(
        "SELECT alias_id FROM aliases WHERE entity_id = ?", (loser,))]
    payload["alias_speaker_ids"] = [r[0] for r in conn.execute(
        "SELECT alias_id FROM aliases WHERE speaker = ?", (loser,))]
    payload["claim_subject_ids"] = [r[0] for r in conn.execute(
        "SELECT claim_id FROM claims WHERE subject_id = ?", (loser,))]
    payload["claim_object_ids"] = [r[0] for r in conn.execute(
        "SELECT claim_id FROM claims WHERE object_id = ?", (loser,))]
    payload["claim_asserted_ids"] = [r[0] for r in conn.execute(
        "SELECT claim_id FROM claims WHERE asserted_by_id = ?", (loser,))]
    payload["review_subject_ids"] = [r[0] for r in conn.execute(
        "SELECT id FROM claim_review_queue WHERE subject_id = ?", (loser,))]

    loser_mentions = conn.execute(
        "SELECT source, row_id, date_iso FROM entity_mentions WHERE entity_id = ?",
        (loser,)).fetchall()
    survivor_pairs = {(r[0], r[1]) for r in conn.execute(
        "SELECT source, row_id FROM entity_mentions WHERE entity_id = ?", (survivor,))}
    payload["mentions"] = [
        {"source": m["source"], "row_id": m["row_id"], "date_iso": m["date_iso"],
         "survivor_had": (m["source"], m["row_id"]) in survivor_pairs}
        for m in loser_mentions
    ]

    ctx_rows = conn.execute(
        "SELECT context_id, entities FROM chunk_contexts WHERE entities LIKE ?",
        (f'%"{loser}"%',)).fetchall()
    payload["contexts"] = []

    loser_ent = conn.execute(
        "SELECT contact_key, first_seen, last_seen FROM entities WHERE entity_id = ?",
        (loser,)).fetchone()
    surv_ent = conn.execute(
        "SELECT contact_key, first_seen, last_seen FROM entities WHERE entity_id = ?",
        (survivor,)).fetchone()
    payload["survivor_contact_key_was_null"] = surv_ent["contact_key"] is None
    payload["survivor_first_seen"] = surv_ent["first_seen"]
    payload["survivor_last_seen"] = surv_ent["last_seen"]

    # -- repoint --
    conn.execute("UPDATE aliases SET entity_id = ? WHERE entity_id = ?", (survivor, loser))
    conn.execute("UPDATE aliases SET speaker = ? WHERE speaker = ?", (survivor, loser))
    conn.execute("UPDATE claims SET subject_id = ? WHERE subject_id = ?", (survivor, loser))
    conn.execute("UPDATE claims SET object_id = ? WHERE object_id = ?", (survivor, loser))
    conn.execute("UPDATE claims SET asserted_by_id = ? WHERE asserted_by_id = ?", (survivor, loser))
    conn.execute("UPDATE claim_review_queue SET subject_id = ? WHERE subject_id = ?", (survivor, loser))

    for m in loser_mentions:
        conn.execute(
            "INSERT OR IGNORE INTO entity_mentions (entity_id, source, row_id, date_iso) "
            "VALUES (?, ?, ?, ?)", (survivor, m["source"], m["row_id"], m["date_iso"]))
    conn.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (loser,))

    for r in ctx_rows:
        ids = json.loads(r["entities"] or "[]")
        payload["contexts"].append({"context_id": r["context_id"], "entities": ids})
        new_ids = list(dict.fromkeys(survivor if x == loser else x for x in ids))
        conn.execute("UPDATE chunk_contexts SET entities = ? WHERE context_id = ?",
                     (json.dumps(new_ids), r["context_id"]))

    # aliases_fts mirrors aliases; a wholesale rebuild is cheap and always consistent.
    _rebuild_aliases_fts(conn)

    # survivor absorbs the loser's seen-window and contact_key (if it had none).
    if loser_ent:
        store.touch_entity_seen(conn, survivor, loser_ent["first_seen"])
        store.touch_entity_seen(conn, survivor, loser_ent["last_seen"])
        if loser_ent["contact_key"] and payload["survivor_contact_key_was_null"]:
            conn.execute("UPDATE entities SET contact_key = ? WHERE entity_id = ?",
                         (loser_ent["contact_key"], survivor))

    # Promote the survivor's title to the best human-readable name now available
    # (a real name/nickname beats an opaque "Unknown (•NNNN)" handle). The id and
    # first_seen provenance are unchanged; the old handle stays as an alias.
    payload["old_canonical"] = _promote_canonical(conn, survivor)

    conn.execute("UPDATE entities SET merged_into = ?, updated_at = ? WHERE entity_id = ?",
                 (survivor, _now(), loser))

    # union contradiction check: undeclared same-subject collisions -> review queue
    # (never auto-invalidated — mirrors the scan's policy; keeps the merge reversible).
    payload["union_review_ids"] = _union_contradiction_check(conn, survivor)

    log_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO merge_log (id, survivor_id, merged_id, reason, created_at, "
        "reverted, payload) VALUES (?, ?, ?, ?, ?, 0, ?)",
        (log_id, survivor, loser, reason, _now(), json.dumps(payload)))
    conn.commit()
    return log_id


def _rebuild_aliases_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM aliases_fts")
    conn.execute(
        "INSERT INTO aliases_fts (surface, norm, entity_id, alias_kind) "
        "SELECT surface, norm, entity_id, kind FROM aliases")


def _promote_canonical(conn: sqlite3.Connection, survivor: str) -> str | None:
    """If the survivor is titled by an opaque handle placeholder, retitle it with
    the best real name/nickname among its aliases. Returns the previous
    canonical_name when changed (for unmerge), else None. ent_me is pinned as "Me".
    """
    if survivor == store.ENT_ME:
        return None
    current = conn.execute(
        "SELECT canonical_name FROM entities WHERE entity_id=?", (survivor,)).fetchone()[0]
    if not _is_opaque_name(current):
        return None
    # Prefer name over nickname, then the longer (more specific) surface.
    ranked = sorted(
        ((0 if r["kind"] == "name" else 1, -len(r["surface"]), r["surface"])
         for r in conn.execute(
             "SELECT DISTINCT surface, kind FROM aliases WHERE entity_id=? "
             "AND kind IN ('name','nickname')", (survivor,))
         if r["surface"] and r["surface"].strip() and not _is_opaque_name(r["surface"])))
    if not ranked or ranked[0][2] == current:
        return None
    conn.execute("UPDATE entities SET canonical_name=? WHERE entity_id=?",
                 (ranked[0][2], survivor))
    return current


def _union_contradiction_check(conn: sqlite3.Connection, survivor: str) -> list[str]:
    """After a merge, two still-valid claims with the same relation but different
    objects may now share the survivor as subject. Queue them for review rather
    than guessing which supersedes which — same-relation/different-object is also
    how legitimate coexisting facts look ("created X" and "created Y")."""
    queued: list[str] = []
    rows = conn.execute(
        "SELECT claim_id, relation, object_text, stated_at FROM claims "
        "WHERE subject_id = ? AND invalidated_at IS NULL ORDER BY stated_at",
        (survivor,)).fetchall()
    by_rel: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by_rel.setdefault(store.norm_surface(r["relation"] or ""), []).append(r)
    for rel, claims in by_rel.items():
        if len(claims) < 2:
            continue
        objs = {store.norm_surface(c["object_text"] or "") for c in claims}
        if len(objs) < 2:
            continue  # same object stated twice — no conflict
        review_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO claim_review_queue (id, thread_key, subject_id, subject_text, "
            "relation, existing_claim_id, new_claim_id, reason, created_at) "
            "VALUES (?, NULL, ?, NULL, ?, ?, ?, ?, ?)",
            (review_id, survivor, claims[0]["relation"], claims[0]["claim_id"],
             claims[-1]["claim_id"], "post-merge same-subject collision", _now()))
        queued.append(review_id)
    return queued


def unmerge(conn: sqlite3.Connection, entity_id: str) -> bool:
    """Reverse the most recent live merge whose loser was `entity_id`. Restores
    every repointed reference and clears the tombstone. Returns False if none."""
    ensure_schema(conn)
    row = conn.execute(
        "SELECT id, survivor_id, merged_id, payload FROM merge_log "
        "WHERE merged_id = ? AND reverted = 0 ORDER BY created_at DESC LIMIT 1",
        (entity_id,)).fetchone()
    if not row:
        return False
    survivor, loser = row["survivor_id"], row["merged_id"]
    payload = json.loads(row["payload"] or "{}")

    if payload.get("alias_ids"):
        conn.executemany("UPDATE aliases SET entity_id = ? WHERE alias_id = ?",
                         [(loser, aid) for aid in payload["alias_ids"]])
    if payload.get("alias_speaker_ids"):
        conn.executemany("UPDATE aliases SET speaker = ? WHERE alias_id = ?",
                         [(loser, aid) for aid in payload["alias_speaker_ids"]])
    if payload.get("claim_subject_ids"):
        conn.executemany("UPDATE claims SET subject_id = ? WHERE claim_id = ?",
                         [(loser, cid) for cid in payload["claim_subject_ids"]])
    if payload.get("claim_object_ids"):
        conn.executemany("UPDATE claims SET object_id = ? WHERE claim_id = ?",
                         [(loser, cid) for cid in payload["claim_object_ids"]])
    if payload.get("claim_asserted_ids"):
        conn.executemany("UPDATE claims SET asserted_by_id = ? WHERE claim_id = ?",
                         [(loser, cid) for cid in payload["claim_asserted_ids"]])
    if payload.get("review_subject_ids"):
        conn.executemany("UPDATE claim_review_queue SET subject_id = ? WHERE id = ?",
                         [(loser, rid) for rid in payload["review_subject_ids"]])

    for m in payload.get("mentions", []):
        conn.execute(
            "INSERT OR IGNORE INTO entity_mentions (entity_id, source, row_id, date_iso) "
            "VALUES (?, ?, ?, ?)", (loser, m["source"], m["row_id"], m["date_iso"]))
        if not m["survivor_had"]:  # this row only reached the survivor via the merge
            conn.execute(
                "DELETE FROM entity_mentions WHERE entity_id = ? AND source = ? AND row_id = ?",
                (survivor, m["source"], m["row_id"]))

    for ctx in payload.get("contexts", []):
        conn.execute("UPDATE chunk_contexts SET entities = ? WHERE context_id = ?",
                     (json.dumps(ctx["entities"]), ctx["context_id"]))

    for rid in payload.get("union_review_ids", []):
        conn.execute("DELETE FROM claim_review_queue WHERE id = ?", (rid,))

    if payload.get("survivor_contact_key_was_null"):
        conn.execute("UPDATE entities SET contact_key = NULL WHERE entity_id = ?", (survivor,))
    if payload.get("old_canonical"):  # undo canonical-name promotion
        conn.execute("UPDATE entities SET canonical_name = ? WHERE entity_id = ?",
                     (payload["old_canonical"], survivor))
    # Shrink the survivor's seen-window back to what it was before it absorbed the loser.
    conn.execute(
        "UPDATE entities SET first_seen = ?, last_seen = ?, updated_at = ? WHERE entity_id = ?",
        (payload.get("survivor_first_seen"), payload.get("survivor_last_seen"), _now(), survivor))

    conn.execute("UPDATE entities SET merged_into = NULL, updated_at = ? WHERE entity_id = ?",
                 (_now(), loser))
    conn.execute("UPDATE merge_log SET reverted = 1 WHERE id = ?", (row["id"],))
    _rebuild_aliases_fts(conn)
    conn.commit()
    return True


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #

def last_run_cutoff(conn: sqlite3.Connection) -> str | None:
    """Timestamp of the last applied merge run — the incremental watermark."""
    ensure_schema(conn)
    row = conn.execute(
        "SELECT MAX(started_at) FROM merge_runs WHERE applied = 1").fetchone()
    return row[0] if row and row[0] else None


@dataclass
class MergeReport:
    run_id: str
    candidates: CandidateSet
    verdicts: list[Verdict]
    merged: list[tuple[str, str, str]] = field(default_factory=list)  # (survivor, loser, reason)
    applied: bool = False


def run_merge(
    conn: sqlite3.Connection, embedder=None, apply: bool = False,
    incremental: bool = False, threshold: float = EMBED_THRESHOLD, client=None,
    user_emails: set | None = None, user_phones: set | None = None,
) -> MergeReport:
    """Full pass: gather candidates, adjudicate, and (when apply) merge + queue.

    Dry-run (apply=False) still adjudicates so verdicts can be inspected, but
    mutates nothing. `user_emails`/`user_phones` gate ent_me absorption; when both
    are None they are discovered from me_identity.
    """
    ensure_schema(conn)
    if user_emails is None and user_phones is None:
        user_emails, user_phones = _user_identity()
    run_id = uuid.uuid4().hex
    started = _now()
    cutoff = last_run_cutoff(conn) if incremental else None
    cands = gather_candidates(conn, embedder, cutoff, threshold, user_emails, user_phones)
    verdicts = adjudicate(conn, cands.adjudicate, client=client)
    report = MergeReport(run_id=run_id, candidates=cands, verdicts=verdicts, applied=apply)

    if not apply:
        return report

    for c in cands.auto:
        survivor, loser = choose_survivor(conn, c.a, c.b)
        merge_entities(conn, survivor, loser, "auto:contact_key", run_id=run_id)
        report.merged.append((survivor, loser, "auto:contact_key"))

    n_unsure = n_distinct = 0
    # ent_me absorption: only when user-identity evidence is present; else distinct.
    for c in cands.identity:
        other = _other_side(c)
        if c.evidence:
            merge_entities(conn, store.ENT_ME, other,
                           ("identity:" + "; ".join(c.evidence))[:200], run_id=run_id)
            report.merged.append((store.ENT_ME, other, "identity"))
        else:
            n_distinct += 1

    for v in verdicts:
        if v.verdict == "merge":
            survivor, loser = choose_survivor(conn, v.candidate.a, v.candidate.b)
            merge_entities(conn, survivor, loser, f"llm:{v.candidate.source}:{v.reason}"[:200],
                           run_id=run_id)
            report.merged.append((survivor, loser, v.reason))
        elif v.verdict == "unsure":
            n_unsure += 1
            _queue_review(conn, run_id, v)
        else:
            n_distinct += 1

    conn.execute(
        "INSERT INTO merge_runs (run_id, started_at, finished_at, cutoff_iso, "
        "n_candidates, n_auto, n_merged, n_unsure, n_distinct, applied) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
        (run_id, started, _now(),
         cutoff, len(cands.auto) + len(cands.identity) + len(cands.adjudicate),
         len(cands.auto), len(report.merged), n_unsure, n_distinct))
    conn.commit()
    return report


def _queue_review(conn: sqlite3.Connection, run_id: str, v: Verdict) -> None:
    conn.execute(
        "INSERT INTO merge_review (id, run_id, entity_a, entity_b, source, score, "
        "reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (uuid.uuid4().hex, run_id, v.candidate.a, v.candidate.b, v.candidate.source,
         v.candidate.score, v.reason, _now()))
