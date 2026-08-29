"""Phase 5 — query-time entity linking + graph retrieval over memory.db.

Pure SQLite + heuristics (no LLM, target <10ms). memory.db is opened read-only
(``file:...?mode=ro``) everywhere so this never contends with the offline
consolidation scan that writes it. The store is expected to be PARTIAL: some
threads scanned, many not yet, the single dominant thread deliberately absent.
Linking degrades gracefully — an absent entity simply doesn't link; nothing
raises.

Kill switch: ``SEMSE_ENTITY_LAYER=0`` (or a missing memory.db) makes
``EntityLayer.load()`` return ``None``, and the search pipeline stays
byte-identical to the pre-entity baseline.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "indexer" / "data"
MEMORY_DB_PATH = DATA_DIR / "memory.db"
CHUNKS_DB_PATH = DATA_DIR / "metadata.db"  # corpus df source (chunks_fts)

ENT_ME = "ent_me"

# Server-computed claim labels (the LLM never does the date arithmetic).
CLAIM_CURRENT = "CURRENT"        # believed true as of `asof`
CLAIM_EXPIRED = "EXPIRED"        # a bounded state whose end date has passed
CLAIM_SUPERSEDED = "SUPERSEDED"  # replaced by a newer claim — history only


def _memory_db_path() -> Path:
    """Effective memory.db path. `SEMSE_MEMORY_DB` overrides the default so evals
    can point the layer at a dev-slice store without touching production."""
    override = os.getenv("SEMSE_MEMORY_DB")
    return Path(override) if override else MEMORY_DB_PATH

# name > nickname > description > handle-only (plan Phase 5). Handle aliases still
# link (sender labels must be reachable) but weakest, so a name match wins.
_KIND_WEIGHT = {"name": 1.0, "nickname": 0.82, "description": 0.6, "handle": 0.5}

# Self-reference tokens route to ent_me. Deliberately excludes we/our (group, not
# self) so "how many times have we…" doesn't drag in the user's whole graph.
_SELF_TOKENS = {"i", "me", "my", "mine", "myself"}
_POSSESSIVE_PRONOUNS = {"her", "his", "their", "its"}

MIN_LINK_SCORE = 0.30
MAX_LINKED = 6
MAX_PER_SPAN = 3          # ambiguity: link at most this many entities per mention
GRAPH_RECENT_SUPERSEDED = 3

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
# "taylor's extension", "jordan's slides", handling curly + straight apostrophes.
_POSSESSIVE_RE = re.compile(r"\b([^\W_]+)['’]s\s+([^\W_]+(?:\s+[^\W_]+)?)", re.I)
_MY_NOUN_RE = re.compile(r"\bmy\s+([^\W_]+(?:\s+[^\W_]+)?)", re.I)
_PRONOUN_NOUN_RE = re.compile(r"\b(her|his|their|its)\s+([^\W_]+(?:\s+[^\W_]+)?)", re.I)

# Function words never accepted as a single-token mention span. Multi-word spans
# may contain them (they're matched as a whole against an alias norm).
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "but", "if", "then", "this", "that", "it", "its",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "they", "them",
    "his", "her", "their", "what", "who", "whom", "when", "where", "why", "how",
    "any", "some", "all", "about", "from", "by", "as", "so", "just", "get", "got",
    "gotten", "go", "going", "went", "say", "said", "talk", "talked", "tell",
    "told", "make", "made", "right", "now", "ever", "still", "back", "much",
    "many", "times", "first", "last", "recent", "recently", "latest", "earliest",
    "call", "called", "up", "come", "up", "did", "does",
}

_HEAD_STOP = _STOP | {"stuff", "thing", "things", "one", "ones"}


def _norm(s: str) -> str:
    """Match indexer.memory_store.norm_surface: NFKC, lowercase, drop punctuation,
    collapse whitespace, keep unicode word chars (so digits/Cyrillic survive)."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(_norm(s))


def _date10(s: str | None) -> str:
    return (s or "")[:10]


# ── result shapes ─────────────────────────────────────────────────────────


@dataclass
class LinkedEntity:
    entity_id: str
    canonical_name: str
    kind: str
    summary: str | None
    contact_key: str | None
    score: float
    spans: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    name_aliases: list[str] = field(default_factory=list)   # surfaces for OR-expansion
    handle_surfaces: list[str] = field(default_factory=list)  # sender labels for filtering
    is_self: bool = False


@dataclass
class AggregateResult:
    operator: str                       # first | last | count | list
    text: str                           # compact structured line for synthesis
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class LinkResult:
    linked: list[LinkedEntity]
    elapsed_ms: float

    @property
    def ids(self) -> list[str]:
        return [e.entity_id for e in self.linked]


@dataclass
class LabeledClaim:
    """A claim about a dossier entity with a SERVER-COMPUTED validity label.
    The synthesis layer never does the date math — it only interprets the label."""
    relation: str
    object_label: str          # object entity's canonical name, else raw object_text
    label: str                 # CLAIM_CURRENT | CLAIM_EXPIRED | CLAIM_SUPERSEDED
    stated_at: str
    valid_until: str | None
    asserted_by_self: bool     # the USER asserted it (hearsay, not the entity's own word)


@dataclass
class EntityDossier:
    """Compact, label-annotated view of one linked entity for the synthesis
    "Known entities" block."""
    entity_id: str
    canonical_name: str
    kind: str
    aliases: list[str]              # top name/nickname surfaces
    claims: list[LabeledClaim]      # most-recent-first, capped
    is_self: bool
    has_own_assertions: bool        # any claim asserted_by THIS entity (non-reply signal)


# ── the layer ─────────────────────────────────────────────────────────────


class EntityLayer:
    """Loaded once at engine init; holds an in-memory snapshot of the (small)
    entity/alias tables plus a corpus-df table for common-word demotion. Live
    reads (claims, mentions, aliases_fts prefix) hit read-only connections."""

    def __init__(self) -> None:
        self.memory_db_path = _memory_db_path()
        self.entities: dict[str, dict] = {}
        self.aliases_by_entity: dict[str, list[dict]] = {}
        self.alias_by_norm: dict[str, list[dict]] = {}
        self.entity_threads: dict[str, set[str]] = {}
        self.df: dict[str, int] = {}
        self.n_chunks: int = 0
        self._load_snapshot()

    # -- construction --

    @classmethod
    def load(cls) -> "EntityLayer | None":
        if os.getenv("SEMSE_ENTITY_LAYER", "1") == "0":
            print("[entities] SEMSE_ENTITY_LAYER=0 — layer disabled", file=sys.stderr)
            return None
        if not _memory_db_path().exists():
            print("[entities] memory.db missing — layer disabled", file=sys.stderr)
            return None
        try:
            layer = cls()
        except sqlite3.Error as e:
            print(f"[entities] memory.db unreadable ({e}) — layer disabled", file=sys.stderr)
            return None
        print(
            f"[entities] layer on: {len(layer.entities)} entities, "
            f"{sum(len(v) for v in layer.aliases_by_entity.values())} aliases, "
            f"df over {layer.n_chunks} chunks",
            file=sys.stderr,
        )
        return layer

    def _open_memory(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.memory_db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_snapshot(self) -> None:
        conn = self._open_memory()
        try:
            for r in conn.execute(
                "SELECT entity_id, kind, canonical_name, summary, contact_key, merged_into "
                "FROM entities"
            ):
                self.entities[r["entity_id"]] = {
                    "kind": r["kind"] or "",
                    "canonical_name": r["canonical_name"] or "",
                    "summary": r["summary"],
                    "contact_key": r["contact_key"],
                    "merged_into": r["merged_into"],
                }
            for r in conn.execute(
                "SELECT entity_id, surface, norm, kind, speaker, source_thread FROM aliases"
            ):
                alias = {
                    "entity_id": r["entity_id"],
                    "surface": r["surface"] or "",
                    "norm": r["norm"] or "",
                    "kind": r["kind"] or "",
                    "speaker": r["speaker"],
                    "source_thread": r["source_thread"],
                }
                self.aliases_by_entity.setdefault(r["entity_id"], []).append(alias)
                self.alias_by_norm.setdefault(alias["norm"], []).append(alias)
                if alias["source_thread"]:
                    self.entity_threads.setdefault(r["entity_id"], set()).add(
                        alias["source_thread"]
                    )
        finally:
            conn.close()
        self._load_df()

    def _load_df(self) -> None:
        """Corpus document-frequency for single-word alias norms (common-word
        demotion). One-time cost at init; bounded by distinct single-word norms."""
        if not CHUNKS_DB_PATH.exists():
            return
        words = {n for n in self.alias_by_norm if n and " " not in n}
        if not words:
            return
        conn = sqlite3.connect(f"file:{CHUNKS_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()
            self.n_chunks = int(row["n"]) if row else 0
            for w in words:
                try:
                    r = conn.execute(
                        "SELECT count(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?",
                        (w,),
                    ).fetchone()
                    self.df[w] = int(r["n"]) if r else 0
                except sqlite3.OperationalError:
                    self.df[w] = 0
        finally:
            conn.close()

    # -- helpers --

    def _survivor(self, entity_id: str) -> str:
        """Follow merged_into tombstones to the surviving entity (cycle-guarded).
        The merge phase lands later; robust to the column being empty now."""
        seen: set[str] = set()
        cur = entity_id
        while cur and cur not in seen:
            seen.add(cur)
            ent = self.entities.get(cur)
            nxt = ent.get("merged_into") if ent else None
            if not nxt or nxt == cur:
                break
            cur = nxt
        return cur

    def _df_weight(self, word: str) -> float:
        """idf-like weight in ~[0.1, 1]: rare alias words ~1, common words demoted."""
        if not self.n_chunks:
            return 1.0
        df = self.df.get(word)
        if not df:
            return 1.0
        w = math.log((self.n_chunks + 1) / (df + 1)) / math.log(self.n_chunks + 1)
        return max(0.1, min(1.0, w))

    def _score(self, kind: str, alias_norm: str, exact: bool, span_norm: str) -> float:
        base = _KIND_WEIGHT.get(kind, 0.4)
        if not exact:
            base *= 0.7
        ntok = span_norm.count(" ") + 1
        base *= 1.0 + 0.15 * (ntok - 1)  # longer spans are more specific
        if " " not in alias_norm:
            base *= self._df_weight(alias_norm)
        return base

    def _match_span(self, conn: sqlite3.Connection, span_norm: str) -> list[dict]:
        """Alias matches for a mention span: exact norm first, then aliases_fts
        prefix (single-word spans only, len>=4) as a fuzzier fallback.

        The prefix must be a genuine LEADING prefix of the alias norm — how
        typing the start of a name behaves. FTS `foo*` also matches interior
        tokens, which over-links generic words ("college" → "foothill college",
        "appointment" → a DMV event); the startswith filter rejects those while
        keeping true head matches ("foothill" → "foothill college")."""
        out: list[dict] = []
        for a in self.alias_by_norm.get(span_norm, []):
            out.append({**a, "exact": True})
        if out:
            return out
        if " " in span_norm or len(span_norm) < 4:
            return out
        try:
            rows = conn.execute(
                "SELECT entity_id, alias_kind, surface, norm FROM aliases_fts "
                "WHERE aliases_fts MATCH ?",
                (f"{span_norm}*",),
            ).fetchall()
        except sqlite3.OperationalError:
            return out
        for r in rows:
            if not (r["norm"] or "").startswith(span_norm):
                continue
            # recover speaker/source_thread from the in-memory alias record
            norm = r["norm"] or ""
            meta = next(
                (a for a in self.alias_by_norm.get(norm, [])
                 if a["entity_id"] == r["entity_id"] and a["kind"] == r["alias_kind"]),
                None,
            )
            out.append({
                "entity_id": r["entity_id"],
                "surface": r["surface"] or "",
                "norm": norm,
                "kind": r["alias_kind"] or "",
                "speaker": meta["speaker"] if meta else None,
                "source_thread": meta["source_thread"] if meta else None,
                "exact": False,
            })
        return out

    # -- span generation --

    def _spans(self, cleaned_query: str, intent_contacts: list[str]) -> list[str]:
        toks = _tokens(cleaned_query)
        spans: list[str] = []
        seen: set[str] = set()

        def add(s: str) -> None:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                spans.append(s)

        n = len(toks)
        for i in range(n):
            # 1-grams: skip function words
            if toks[i] not in _STOP:
                add(toks[i])
            if i + 1 < n:
                add(f"{toks[i]} {toks[i+1]}")
            if i + 2 < n:
                add(f"{toks[i]} {toks[i+1]} {toks[i+2]}")
        for c in intent_contacts or []:
            add(_norm(c))
        return spans

    def _possessive_pairs(self, raw_query: str) -> list[tuple[str, str]]:
        """(possessor_span, head_noun) pairs. Possessor 'my' → the self marker."""
        pairs: list[tuple[str, str]] = []
        for m in _POSSESSIVE_RE.finditer(raw_query):
            pairs.append((_norm(m.group(1)), _norm(m.group(2))))
        for m in _MY_NOUN_RE.finditer(raw_query):
            pairs.append((ENT_ME, _norm(m.group(1))))
        return pairs

    # -- the linker --

    def link(
        self,
        cleaned_query: str,
        raw_query: str,
        intent_contacts: list[str] | None,
        asof: str | None,
    ) -> LinkResult:
        t0 = time.perf_counter()
        conn = self._open_memory()
        try:
            cand: dict[str, dict] = {}
            per_span: dict[str, list[tuple[str, float]]] = {}
            anchored_threads: set[str] = set()
            deferred_desc: list[tuple[str, dict, float]] = []

            for span in self._spans(cleaned_query, intent_contacts or []):
                for m in self._match_span(conn, span):
                    eid = self._survivor(m["entity_id"])
                    if eid not in self.entities:
                        continue
                    score = self._score(m["kind"], m["norm"], m["exact"], span)
                    if m["kind"] == "description":
                        deferred_desc.append((span, {**m, "entity_id": eid}, score))
                        continue
                    per_span.setdefault(span, []).append((eid, score))
                    self._bump(cand, eid, score, span, f"{m['kind']}:{m['norm']}")
                    if m["source_thread"]:
                        anchored_threads.add(m["source_thread"])

            # Threads of already-linked entities also count as anchored.
            for eid in list(cand):
                anchored_threads |= self.entity_threads.get(eid, set())

            # Description aliases link only when their source_thread's participant
            # is independently anchored ("the game" alone links nothing).
            for span, m, score in deferred_desc:
                thread_ok = m["source_thread"] and m["source_thread"] in anchored_threads
                speaker_ok = m["speaker"] and self._survivor(m["speaker"]) in cand
                if thread_ok or speaker_ok:
                    per_span.setdefault(span, []).append((m["entity_id"], score))
                    self._bump(cand, m["entity_id"], score, span, f"description:{m['norm']}")

            # Ambiguity: cap each mention span to its top-N entities (never fail).
            keep: set[str] = set()
            for matches in per_span.values():
                for eid, _ in sorted(matches, key=lambda x: -x[1])[:MAX_PER_SPAN]:
                    keep.add(eid)
            for eid in list(cand):
                if eid not in keep:
                    del cand[eid]

            # Self-reference → ent_me.
            if ENT_ME in self.entities and self._has_self_ref(cleaned_query):
                self._bump(cand, ENT_ME, 0.7, "<self>", "self-reference", is_self=True)

            # Possessive one-hop: possessor's claim neighborhood scored vs head noun.
            for possessor, head in self._possessive_pairs(raw_query):
                self._possessive_hop(conn, cand, possessor, head, asof, intent_contacts)
            self._pronoun_hop(conn, cand, raw_query, asof)

            linked = self._finalize(cand)
        finally:
            conn.close()
        return LinkResult(linked=linked, elapsed_ms=(time.perf_counter() - t0) * 1000)

    def _bump(
        self,
        cand: dict[str, dict],
        eid: str,
        score: float,
        span: str,
        reason: str,
        is_self: bool = False,
    ) -> None:
        cur = cand.get(eid)
        if cur is None:
            cand[eid] = {"score": score, "spans": [span], "reasons": [reason], "is_self": is_self}
            return
        cur["score"] = max(cur["score"], score)
        if span not in cur["spans"]:
            cur["spans"].append(span)
        if reason not in cur["reasons"]:
            cur["reasons"].append(reason)
        cur["is_self"] = cur["is_self"] or is_self

    def _has_self_ref(self, cleaned_query: str) -> bool:
        return bool(_SELF_TOKENS & set(_tokens(cleaned_query)))

    def _resolve_possessor(
        self,
        conn: sqlite3.Connection,
        possessor: str,
        intent_contacts: list[str] | None,
    ) -> list[str]:
        if possessor == ENT_ME:
            return [ENT_ME] if ENT_ME in self.entities else []
        matches = self._match_span(conn, possessor)
        scored: dict[str, float] = {}
        for m in matches:
            if m["kind"] == "description":
                continue
            eid = self._survivor(m["entity_id"])
            if eid in self.entities:
                scored[eid] = max(scored.get(eid, 0.0), self._score(m["kind"], m["norm"], m["exact"], possessor))
        return [eid for eid, _ in sorted(scored.items(), key=lambda x: -x[1])[:2]]

    def _possessive_hop(
        self,
        conn: sqlite3.Connection,
        cand: dict[str, dict],
        possessor: str,
        head: str,
        asof: str | None,
        intent_contacts: list[str] | None,
    ) -> None:
        head_toks = {t for t in _tokens(head) if t not in _HEAD_STOP}
        if not head_toks:
            return
        for p_eid in self._resolve_possessor(conn, possessor, intent_contacts):
            # Possessor itself links when named in the query.
            if possessor != ENT_ME:
                self._bump(cand, p_eid, 0.6, possessor, "possessive-owner")
            for n_eid, conn_tokens, conf, stated in self._claim_neighbors(conn, p_eid, asof):
                overlap = self._head_overlap(n_eid, head_toks, conn_tokens)
                if overlap > 0:
                    score = 0.55 + 0.1 * overlap + 0.05 * (conf or 0.0)
                    self._bump(cand, n_eid, score, f"{possessor}'s {head}",
                               f"possessive-hop({overlap})")

    def _pronoun_hop(
        self,
        conn: sqlite3.Connection,
        cand: dict[str, dict],
        raw_query: str,
        asof: str | None,
    ) -> None:
        """'her/his/their <noun>' → hop from the single strongest linked person
        entity (light coreference; skipped when the antecedent is ambiguous)."""
        m = _PRONOUN_NOUN_RE.search(raw_query)
        if not m:
            return
        head = _norm(m.group(2))
        head_toks = {t for t in _tokens(head) if t not in _HEAD_STOP}
        if not head_toks:
            return
        persons = sorted(
            (eid for eid in cand if self.entities.get(eid, {}).get("kind") == "person"
             and eid != ENT_ME),
            key=lambda e: -cand[e]["score"],
        )
        if len(persons) != 1:
            return
        p_eid = persons[0]
        for n_eid, conn_tokens, conf, stated in self._claim_neighbors(conn, p_eid, asof):
            overlap = self._head_overlap(n_eid, head_toks, conn_tokens)
            if overlap > 0:
                self._bump(cand, n_eid, 0.55 + 0.1 * overlap, f"{m.group(1)} {head}",
                           f"pronoun-hop({overlap})")

    def _claim_neighbors(
        self,
        conn: sqlite3.Connection,
        possessor: str,
        asof: str | None,
    ) -> list[tuple[str, set[str], float, str]]:
        """Neighbor entities reachable in one claim hop from `possessor`, with the
        connecting claim's relation+object_text tokens for head-noun scoring."""
        rows = conn.execute(
            "SELECT subject_id, object_id, relation, object_text, confidence, stated_at "
            "FROM claims WHERE subject_id = ? OR object_id = ? OR asserted_by_id = ?",
            (possessor, possessor, possessor),
        ).fetchall()
        out: list[tuple[str, set[str], float, str]] = []
        for r in rows:
            if asof and _date10(r["stated_at"]) > _date10(asof):
                continue
            neighbor = None
            if r["subject_id"] == possessor and r["object_id"]:
                neighbor = self._survivor(r["object_id"])
            elif r["object_id"] == possessor and r["subject_id"]:
                neighbor = self._survivor(r["subject_id"])
            if not neighbor or neighbor == possessor or neighbor not in self.entities:
                continue
            conn_tokens = set(_tokens(r["relation"] or "")) | set(_tokens(r["object_text"] or ""))
            out.append((neighbor, conn_tokens, r["confidence"] or 0.0, r["stated_at"] or ""))
        return out

    def _head_overlap(self, eid: str, head_toks: set[str], conn_tokens: set[str]) -> int:
        ent = self.entities.get(eid, {})
        bag = set(_tokens(ent.get("kind", ""))) | set(_tokens(ent.get("canonical_name", "")))
        bag |= set(_tokens(ent.get("summary") or ""))
        for a in self.aliases_by_entity.get(eid, []):
            bag |= set(_tokens(a["norm"]))
        bag |= conn_tokens
        return len(head_toks & bag)

    def _finalize(self, cand: dict[str, dict]) -> list[LinkedEntity]:
        picked = [
            (eid, c) for eid, c in cand.items()
            if c["score"] >= MIN_LINK_SCORE or c["is_self"]
        ]
        picked.sort(key=lambda x: -x[1]["score"])
        out: list[LinkedEntity] = []
        for eid, c in picked[:MAX_LINKED]:
            ent = self.entities[eid]
            names, handles = [], []
            for a in self.aliases_by_entity.get(eid, []):
                if a["kind"] in ("name", "nickname") and a["surface"] not in names:
                    names.append(a["surface"])
                elif a["kind"] == "handle" and a["surface"] not in handles:
                    handles.append(a["surface"])
            out.append(LinkedEntity(
                entity_id=eid,
                canonical_name=ent["canonical_name"],
                kind=ent["kind"],
                summary=ent["summary"],
                contact_key=ent["contact_key"],
                score=round(c["score"], 3),
                spans=c["spans"],
                reasons=c["reasons"],
                name_aliases=names[:6],
                handle_surfaces=handles[:8],
                is_self=c["is_self"],
            ))
        return out

    # ── graph retriever ────────────────────────────────────────────────────

    def graph_chunks(self, linked_ids: list[str], asof: str | None, k: int) -> list[str]:
        """Currently-valid claims of linked entities (+ the 3 most recently
        superseded, for transition questions) → evidence row_ids → chunk_ids.

        ASOF discipline: exclude claims stated after asof; evaluate validity as of
        asof (a supersession that happens after asof hasn't occurred yet), never
        against the wall clock. Mirrors the retrieval before-bound in search()."""
        if not linked_ids:
            return []
        conn = self._open_memory()
        try:
            ph = ",".join("?" * len(linked_ids))
            rows = conn.execute(
                f"SELECT claim_id, stated_at, invalidated_at, superseded_by, confidence, evidence "
                f"FROM claims WHERE subject_id IN ({ph}) OR object_id IN ({ph})",
                linked_ids + linked_ids,
            ).fetchall()
            asof10 = _date10(asof) if asof else None
            current: list[sqlite3.Row] = []
            superseded: list[sqlite3.Row] = []
            for r in rows:
                if asof10 and _date10(r["stated_at"]) > asof10:
                    continue  # future claim — doesn't exist yet at asof
                if asof10:
                    is_current = (not r["invalidated_at"]) or _date10(r["invalidated_at"]) > asof10
                else:
                    is_current = (not r["invalidated_at"]) and (not r["superseded_by"])
                (current if is_current else superseded).append(r)

            current.sort(key=lambda r: ((r["confidence"] or 0.0), r["stated_at"] or ""), reverse=True)
            superseded.sort(key=lambda r: (r["stated_at"] or ""), reverse=True)
            ordered = current + superseded[:GRAPH_RECENT_SUPERSEDED]

            # Evidence (source,row_id) in claim order → chunk_ids via rowid_chunk_map.
            refs: list[tuple[str, int]] = []
            for r in ordered:
                try:
                    ev = json.loads(r["evidence"] or "[]")
                except (TypeError, ValueError):
                    continue
                for e in ev:
                    src, rid = e.get("source"), e.get("row_id")
                    if src is not None and rid is not None:
                        refs.append((src, int(rid)))
            return self._rows_to_chunks(conn, refs, k)
        finally:
            conn.close()

    def _rows_to_chunks(
        self, conn: sqlite3.Connection, refs: list[tuple[str, int]], k: int
    ) -> list[str]:
        """Map (source,row_id) refs to chunk_ids, preserving order and de-duping."""
        if not refs:
            return []
        wanted = {(s, r) for s, r in refs}
        ph = ",".join("?" * len(wanted))
        params = [v for pair in wanted for v in pair]
        lookup: dict[tuple[str, int], str] = {}
        cur = conn.execute(
            f"SELECT source, row_id, chunk_id FROM rowid_chunk_map "
            f"WHERE (source, row_id) IN (VALUES {','.join(['(?,?)'] * len(wanted))})",
            params,
        )
        for row in cur:
            lookup[(row["source"], row["row_id"])] = row["chunk_id"]
        ordered_ids: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            cid = lookup.get(ref)
            if cid and cid not in seen:
                seen.add(cid)
                ordered_ids.append(cid)
            if len(ordered_ids) >= k:
                break
        return ordered_ids

    # ── aggregate-operator routing ───────────────────────────────────────────

    def aggregate(
        self, raw_query: str, linked: list[LinkedEntity], asof: str | None
    ) -> AggregateResult | None:
        """Generic operators (first/last/count/list), computed by SQL over
        entity_mentions + claims. Returns None when no operator fires or nothing
        is linked (graceful degradation)."""
        op = self._detect_operator(raw_query)
        if not op or not linked:
            return None
        ids = [e.entity_id for e in linked]
        names = ", ".join(dict.fromkeys(e.canonical_name for e in linked))
        conn = self._open_memory()
        try:
            if op == "count":
                return self._agg_count(conn, ids, names, asof)
            if op in ("first", "last"):
                return self._agg_extremum(conn, ids, names, asof, op)
            if op == "list":
                return self._agg_list(conn, ids, names, asof)
        finally:
            conn.close()
        return None

    @staticmethod
    def _detect_operator(query: str) -> str | None:
        return detect_aggregate_operator(query)

    def _mention_dates(
        self, conn: sqlite3.Connection, ids: list[str], asof: str | None
    ) -> list[sqlite3.Row]:
        ph = ",".join("?" * len(ids))
        params: list = list(ids)
        clause = ""
        if asof:
            clause = " AND substr(date_iso,1,10) <= ?"
            params.append(_date10(asof))
        return conn.execute(
            f"SELECT source, row_id, date_iso FROM entity_mentions "
            f"WHERE entity_id IN ({ph}){clause} ORDER BY date_iso ASC",
            params,
        ).fetchall()

    def _agg_count(self, conn, ids, names, asof) -> AggregateResult | None:
        rows = self._mention_dates(conn, ids, asof)
        if not rows:
            return None
        days = sorted({_date10(r["date_iso"]) for r in rows if r["date_iso"]})
        first, last = days[0], days[-1]
        # Evidence: earliest + latest + evenly-spaced middle samples.
        picks = [rows[0], rows[-1]]
        if len(rows) > 4:
            picks += [rows[len(rows) // 3], rows[2 * len(rows) // 3]]
        refs = [(r["source"], int(r["row_id"])) for r in picks]
        text = (
            f"[computed from the memory index — authoritative for coverage] "
            f"{names} referenced on {len(days)} distinct day(s) between {first} and {last}. "
            f"Top excerpts show only a few of those days; do not imply a small total."
        )
        return AggregateResult("count", text, self._rows_to_chunks(conn, refs, 6))

    def _agg_extremum(self, conn, ids, names, asof, op) -> AggregateResult | None:
        rows = self._mention_dates(conn, ids, asof)
        if not rows:
            return None
        r = rows[0] if op == "first" else rows[-1]
        when = _date10(r["date_iso"])
        word = "earliest" if op == "first" else "most recent"
        text = (
            f"[computed from the memory index — authoritative] {word} reference to "
            f"{names} on record: {when}."
        )
        refs = [(r["source"], int(r["row_id"]))]
        return AggregateResult(op, text, self._rows_to_chunks(conn, refs, 3))

    def _agg_list(self, conn, ids, names, asof) -> AggregateResult | None:
        ph = ",".join("?" * len(ids))
        params: list = list(ids)
        clause = ""
        if asof:
            clause = " AND substr(stated_at,1,10) <= ?"
            params.append(_date10(asof))
        rows = conn.execute(
            f"SELECT DISTINCT relation, object_text, object_id, evidence FROM claims "
            f"WHERE subject_id IN ({ph}){clause} AND object_text IS NOT NULL",
            params,
        ).fetchall()
        if not rows:
            return None
        items: list[str] = []
        refs: list[tuple[str, int]] = []
        for r in rows:
            label = r["object_text"]
            if r["object_id"] and r["object_id"] in self.entities:
                label = self.entities[r["object_id"]]["canonical_name"] or label
            if label and label not in items:
                items.append(label)
            try:
                for e in json.loads(r["evidence"] or "[]"):
                    if e.get("source") is not None and e.get("row_id") is not None:
                        refs.append((e["source"], int(e["row_id"])))
            except (TypeError, ValueError):
                pass
        text = (
            f"[computed from the memory index] distinct facts on record about {names}: "
            + "; ".join(items[:12]) + "."
        )
        return AggregateResult("list", text, self._rows_to_chunks(conn, refs, 6))

    # ── entity dossiers (labeled claims for synthesis) ───────────────────────

    def dossiers(
        self,
        linked: list[LinkedEntity],
        asof: str | None,
        max_entities: int = MAX_LINKED,
        max_claims: int = 10,
    ) -> list[EntityDossier]:
        """Label-annotated dossiers for the top linked entities. Validity labels
        (CURRENT/EXPIRED/SUPERSEDED) are computed here with asof discipline; the
        LLM only interprets them. Returns [] when nothing linked (graceful)."""
        if not linked:
            return []
        import datetime as _dt
        asof10 = _date10(asof) if asof else None
        today10 = asof10 or _dt.date.today().isoformat()
        conn = self._open_memory()
        try:
            return [
                self._dossier(conn, e, asof10, today10, max_claims)
                for e in linked[:max_entities]
            ]
        finally:
            conn.close()

    def _dossier(self, conn, e: LinkedEntity, asof10, today10, max_claims) -> EntityDossier:
        rows = conn.execute(
            "SELECT claim_id, relation, object_text, object_id, asserted_by_id, "
            "stated_at, valid_until, invalidated_at, superseded_by "
            "FROM claims WHERE subject_id = ?",
            (e.entity_id,),
        ).fetchall()
        # supersession under asof: a supersession dated after asof hasn't happened
        # yet, so the original is still live — need each claim's stated date.
        stated_of = {r["claim_id"]: _date10(r["stated_at"]) for r in rows}
        labeled: list[dict] = []
        for r in rows:
            stated = _date10(r["stated_at"])
            if asof10 and stated > asof10:
                continue  # future claim — doesn't exist yet at asof
            labeled.append({
                "relation": r["relation"] or "",
                "object_label": self._object_label(r),
                "label": self._claim_label(r, asof10, today10, stated_of),
                "stated_at": stated,
                "valid_until": _date10(r["valid_until"]) if r["valid_until"] else None,
                "asserted_by_self": r["asserted_by_id"] == ENT_ME,
            })
        self._reconcile_co_asserted(labeled, today10)
        labeled.sort(key=lambda c: c["stated_at"], reverse=True)
        has_own = conn.execute(
            "SELECT 1 FROM claims WHERE asserted_by_id = ? LIMIT 1", (e.entity_id,)
        ).fetchone() is not None
        return EntityDossier(
            entity_id=e.entity_id,
            canonical_name=e.canonical_name,
            kind=e.kind,
            aliases=e.name_aliases[:4],
            claims=[LabeledClaim(**c) for c in labeled[:max_claims]],
            is_self=e.is_self,
            has_own_assertions=has_own,
        )

    def _object_label(self, r) -> str:
        oid = r["object_id"]
        if oid and oid in self.entities:
            name = self.entities[oid].get("canonical_name")
            if name:
                return name
        return r["object_text"] or ""

    def _claim_label(self, r, asof10, today10, stated_of) -> str:
        sup = r["superseded_by"]
        if sup:
            sup_date = stated_of.get(sup)
            if not asof10 or (sup_date and sup_date <= asof10):
                return CLAIM_SUPERSEDED
        inval = _date10(r["invalidated_at"]) if r["invalidated_at"] else None
        if inval and (not asof10 or inval <= asof10):
            return CLAIM_SUPERSEDED
        vu = _date10(r["valid_until"]) if r["valid_until"] else None
        if vu and vu < today10:
            return CLAIM_EXPIRED
        return CLAIM_CURRENT

    def _reconcile_co_asserted(self, labeled: list[dict], today10: str) -> None:
        """A broad unbounded claim co-asserted (same day, same relation) with a
        bounded one inherits that bound. Closes the "in Asia (unbounded) + in
        China (until 7/15)" gap: both describe one state, so the tightest stated
        end governs. Scoped to a single (relation, stated day) — a genuinely
        later re-assertion keeps its own (missing) bound and stays CURRENT."""
        bound: dict[tuple[str, str], str] = {}
        for c in labeled:
            if c["valid_until"]:
                key = (c["relation"], c["stated_at"])
                prev = bound.get(key)
                bound[key] = min(prev, c["valid_until"]) if prev else c["valid_until"]
        for c in labeled:
            if c["valid_until"] is None and c["label"] == CLAIM_CURRENT:
                vu = bound.get((c["relation"], c["stated_at"]))
                if vu:
                    c["valid_until"] = vu
                    if vu < today10:
                        c["label"] = CLAIM_EXPIRED


def detect_aggregate_operator(query: str) -> str | None:
    """Generic aggregate operators (not content categories): first/last/count/list."""
    q = f" {query.lower().strip()} "
    if re.search(r"\bhow (many|often|much)\b|\bnumber of\b|\bcount\b", q):
        return "count"
    if re.search(r"\bfirst\b|\bearliest\b|\bbegin|\bstart(ed)?\b", q):
        return "first"
    if re.search(r"\blast\b|\bmost recent\b|\blatest\b", q):
        return "last"
    if re.search(r"\ball\b|\bevery\b|\blist\b|\bwhat .*\b(come up|come out)\b|"
                 r"\bwhat .*\bhave\b", q):
        return "list"
    return None


def log_link(query: str, result: LinkResult) -> None:
    """Stderr trace of link decisions, styled after the existing [search] intent log."""
    if not result.linked:
        print(f"[entities] query=\"{query}\" linked nothing "
              f"({result.elapsed_ms:.1f}ms)", file=sys.stderr)
        return
    parts = [
        f"{e.canonical_name}({e.kind},s={e.score},{'|'.join(e.reasons)})"
        for e in result.linked
    ]
    print(f"[entities] query=\"{query}\" linked {len(result.linked)} in "
          f"{result.elapsed_ms:.1f}ms: {'; '.join(parts)}", file=sys.stderr)
