"""Search pipeline: hybrid (FAISS + FTS5) retrieval → strict-prompt synthesis."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from openai import AsyncOpenAI

from . import hyperspell, temporal
from .models import ChunkMessage, QueryIntent, SearchRequest, SearchResponse, SourceResult

INDEXER_DIR = Path(__file__).resolve().parent.parent / "indexer"
sys.path.insert(0, str(INDEXER_DIR))
from embedder import Embedder  # noqa: E402

DATA_DIR = INDEXER_DIR / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
META_DB_PATH = DATA_DIR / "metadata.db"
ID_MAP_PATH = DATA_DIR / "id_map.json"
IMAGE_INDEX_PATH = DATA_DIR / "images.faiss"
IMAGE_ID_MAP_PATH = DATA_DIR / "image_id_map.json"

OPENAI_MODEL = "gpt-4o-mini"

# Strict prompt: model must return empty string when excerpts don't address the query.
SYSTEM_PROMPT = """You are a personal memory search assistant.

The user has typed a query into a search bar and you receive conversation excerpts retrieved from their messages and emails. Your job is to decide whether to surface a synthesized answer or stay silent.

DEFAULT BEHAVIOR — STAY SILENT. Most queries are lookups ("foothill canvas", "rent payment", "ryan birthday"). For these, return NOTHING — not the word "empty", not "no answer", not an apology. Just a literal empty response. The source list below your answer is what the user will read.

ONLY answer when ALL of these are true:
- The query is phrased as a question (ends in "?", or starts with "what/who/where/when/why/how/does/has/is/can/should/did/will/tell me/explain/summarize").
- At least one excerpt directly answers that question with specific facts.

When you DO answer:
- 1–2 sentences max. No preamble like "Based on the excerpts…". No restating the question.
- Never quote a single message verbatim as if it were the answer. Synthesize.
- Never define or explain terms from the query (don't explain what "Foothill" is — find conversations).
- Cite sources inline as [iMessage·<Name>] or [Mail·<Name>]. Prefer the contact-list name; if a sender is labeled "Unknown (•1234)" but introduces themselves in the message body ("hi I'm Conor"), it's fine to use the body-mentioned name in your answer. Never invent a name with no textual basis.
- Senders are tagged "(new contact, not in address book)" when the user has never saved their number. For "who did I just meet?" / "who introduced themselves to me?" only count people with that tag — existing contacts saying "great to meet you" are not new acquaintances.
- A "Known contacts in this excerpt set" block may appear above the excerpts. Those are pre-built relationship summaries (cluster-sampled across the user's history). Treat them as factual context. If the user asks "what's my relationship with X" or "what does X do" or similar, the summary IS the right answer — synthesize from it directly, don't refuse for lack of an excerpt that literally says it.

SPEAKER ATTRIBUTION — CRITICAL:
Each excerpt line is formatted "<SpeakerName>: <message text>". The speaker label IS authoritative — the prefix tells you exactly who wrote that line.
- Lines starting with "Me:" are the USER's own words. Never attribute them to anyone else.
- For questions like "what does X think about Y", "does X prefer A or B", "did X mention Z": count ONLY lines where the prefix is X. Lines prefixed "Me:" cannot answer questions about what X said, thought, or prefers — they're the user, not X.
- If the excerpts only contain "Me:" lines on the topic (the user opining, the contact silent), stay silent. The user already knows what they themselves said.
- When citing a fact, mentally check: "is the source line prefixed with the right speaker?" If not, do not include that fact.

To stay silent, your entire response must be the empty string. Do not say "no answer", "EMPTY", "(nothing)", or anything else."""


# Separate prompt used only when the query names a specific contact. The
# excerpts are pre-filtered to ONLY that contact's lines — no user messages,
# no other speakers. We use a tighter prompt here because the more complex
# instructions in SYSTEM_PROMPT (with [EVIDENCE]/[CONTEXT] tagging rules)
# kept confusing the model into attributing user words to the contact.
CONTACT_FOCUSED_PROMPT = """You are analyzing what one specific person said, based on excerpts of their own messages.

The excerpts contain ONLY the targeted person's own messages. The user's questions and opinions are NOT shown to you. Every line you see was sent BY the targeted person.

How to answer (1–2 sentences):
- Describe what the person seems to think/prefer/like by PARAPHRASING patterns across their messages. Mention concrete topics, attitudes, or recurring themes.
- If you use double quotation marks ("…"), the text inside MUST appear verbatim in the messages below. A single fabricated quote invalidates the entire answer. When in doubt, paraphrase without quotes — that's always safe.
- Apostrophes for possessives/contractions are fine. Single quotes around phrases (' … ') are not used for quotation here — use double quotes if you must quote.

Empty-response cases:
- If the messages don't contain enough information to answer, return an empty string.
- Don't guess, don't extrapolate from the question itself, don't fill in plausible-sounding content.

Output format:
- 1–2 sentences. No preamble. No "Based on the messages…". No apologies. No restating the question.
- To stay silent: return a literal empty string. Don't write "no answer" or anything similar."""

RRF_K = 60  # standard reciprocal rank fusion constant
FUSE_FETCH = 24  # how many candidates each retriever pulls before fusion

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
_NAME_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _query_tokens(query: str) -> set[str]:
    return {t.lower() for t in _QUERY_TOKEN_RE.findall(query)}


def _normalize_name(name: str) -> str:
    """Lowercase + strip punctuation. Used to fuzzy-resolve LLM-extracted
    contact names against the index's canonical names."""
    if not name:
        return ""
    return _NAME_NORM_RE.sub("", name.lower()).strip()


def _filter_for_target_contacts(
    messages: list[ChunkMessage],
    target: set[str],
) -> list[ChunkMessage]:
    """Keep ONLY messages spoken by target contacts. No surrounding context.

    Previous versions kept one preceding line for context, but the LLM kept
    paraphrasing that preceding line as the contact's view. With user lines
    gone entirely, there's nothing for it to mistakenly attribute. The
    contact's tone, vocabulary, and topical pattern is still fully present
    in their own messages."""
    if not messages or not target:
        return messages
    return [m for m in messages if m.sender in target]


# Tokens we never accept as a contact candidate in the fallback scan, even if
# they happen to appear in the contact index. Keeps the heuristic from
# attaching meaning to verbs/articles/etc that share spelling with nicknames.
_CONTACT_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "but", "if", "then", "this", "that", "it",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "they",
    "what", "who", "when", "where", "why", "how", "any", "some", "all",
    "about", "from", "by", "as", "tell", "show", "find", "get", "got",
    "say", "said", "ask", "asked", "think", "thought", "like", "likes",
    "liked", "prefer", "prefers", "preferred", "want", "wants", "wanted",
    "claude", "chatgpt", "gpt", "openai", "anthropic", "lockheed", "amazon",
    "google", "microsoft", "stanford", "harvard", "mit", "berkeley",
    "mail", "email", "text", "texts", "message", "messages", "picture",
    "photo", "image", "pic",
}


def _best_message_index(messages: list[ChunkMessage], q_tokens: set[str]) -> int:
    """Pick the message with the highest query-token overlap. Tie-breaks to the
    longest message (more substantive)."""
    if not messages or not q_tokens:
        return 0
    best_i, best_score, best_len = 0, -1, -1
    for i, m in enumerate(messages):
        toks = {t.lower() for t in _QUERY_TOKEN_RE.findall(m.text or "")}
        score = len(toks & q_tokens)
        text_len = len(m.text or "")
        if score > best_score or (score == best_score and text_len > best_len):
            best_i, best_score, best_len = i, score, text_len
    return best_i


class SearchEngine:
    def __init__(self) -> None:
        if not INDEX_PATH.exists() or not META_DB_PATH.exists():
            raise FileNotFoundError(
                f"Index or metadata not found. Run `python indexer/build_index.py` first.\n"
                f"  expected: {INDEX_PATH}\n  expected: {META_DB_PATH}"
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        self.id_map: list[str] = json.loads(ID_MAP_PATH.read_text())
        self.embedder = Embedder()
        self._db_path = str(META_DB_PATH)
        self._openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Image index is optional — if it doesn't exist, image search is skipped.
        self.image_index = None
        self.image_id_map: list[int] = []
        self.clip_embedder = None
        if IMAGE_INDEX_PATH.exists() and IMAGE_ID_MAP_PATH.exists():
            self.image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
            self.image_id_map = json.loads(IMAGE_ID_MAP_PATH.read_text())
        self._contact_summaries: dict[str, dict] = self._load_contact_summaries()
        self._contact_norm_index: dict[str, set[str]] = self._build_contact_norm_index()

    def _build_contact_norm_index(self) -> dict[str, set[str]]:
        """Map normalized name token → set of canonical contact names that
        contain that token. Used to fuzzy-resolve an LLM-extracted contact
        ("jerry") to canonical names from the index ("Jerry Yan").
        """
        index: dict[str, set[str]] = {}
        with self._open_db() as conn:
            try:
                rows = conn.execute("SELECT DISTINCT contact_names FROM chunks").fetchall()
            except sqlite3.OperationalError:
                return index
        seen_canonical: set[str] = set()
        for r in rows:
            try:
                names = json.loads(r["contact_names"])
            except (TypeError, ValueError):
                continue
            for canonical in names:
                if not isinstance(canonical, str) or canonical in seen_canonical:
                    continue
                seen_canonical.add(canonical)
                # Index by each whitespace-separated token AND by the full
                # normalized name so multi-word LLM extractions still match.
                full_norm = _normalize_name(canonical)
                if full_norm:
                    index.setdefault(full_norm, set()).add(canonical)
                for tok in full_norm.split():
                    if len(tok) >= 2:
                        index.setdefault(tok, set()).add(canonical)
        return index

    def _extract_contacts_from_query(self, query: str) -> set[str]:
        """Heuristic fallback when the LLM misses a contact name that's in
        the query verbatim. We scan tokens (length >= 2) against the
        normalized contact index. Stopwords are skipped so common words
        don't accidentally resolve to a contact named e.g. "and"."""
        if not query or not self._contact_norm_index:
            return set()
        tokens = _QUERY_TOKEN_RE.findall(query.lower())
        # Filter out generic English stopwords + verbs that often appear
        # alongside contact names. (Don't filter on length alone — short
        # nicknames like "M" or "IL" exist in the user's index.)
        tokens = [t for t in tokens if t not in _CONTACT_QUERY_STOPWORDS]
        out: set[str] = set()
        for t in tokens:
            if t in self._contact_norm_index:
                # Only accept tokens that map UNAMBIGUOUSLY (one canonical name)
                # or where the token IS the full canonical name. Otherwise a
                # common first name like "alex" could pull in many people.
                candidates = self._contact_norm_index[t]
                if len(candidates) == 1:
                    out.update(candidates)
        return out

    def _resolve_contacts(self, extracted: list[str]) -> set[str]:
        """Given LLM-extracted contact strings, return the set of canonical
        contact names from the index they could refer to. Multi-token strings
        require all tokens to point to the same canonical name (intersection)."""
        resolved: set[str] = set()
        for raw in extracted:
            norm = _normalize_name(raw)
            if not norm:
                continue
            # Exact normalized hit first
            if norm in self._contact_norm_index:
                resolved.update(self._contact_norm_index[norm])
                continue
            tokens = [t for t in norm.split() if len(t) >= 2]
            if not tokens:
                continue
            # Intersect per-token candidate sets so "jerry yan" prefers
            # contacts that match BOTH tokens; if intersection is empty,
            # fall back to the union (best-effort soft match).
            per_token = [self._contact_norm_index.get(t, set()) for t in tokens]
            intersect = set.intersection(*per_token) if per_token else set()
            if intersect:
                resolved.update(intersect)
            else:
                for s in per_token:
                    resolved.update(s)
        return resolved

    def _load_contact_summaries(self) -> dict[str, dict]:
        with self._open_db() as conn:
            try:
                rows = conn.execute(
                    "SELECT name, summary, total_chunks, last_30d_chunks, "
                    "first_message_iso, last_message_iso FROM contact_summaries"
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
        out: dict[str, dict] = {}
        for r in rows:
            out[r["name"]] = {
                "summary": r["summary"],
                "total_chunks": r["total_chunks"],
                "last_30d_chunks": r["last_30d_chunks"],
                "first": (r["first_message_iso"] or "")[:10],
                "last": (r["last_message_iso"] or "")[:10],
            }
        return out

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- Retrievers (each returns ordered list of chunk_ids) --

    def _dense_search(self, query: str, k: int) -> list[str]:
        vec = self.embedder.embed_one(query).astype("float32").reshape(1, -1)
        _scores, idxs = self.index.search(vec, k)
        return [self.id_map[i] for i in idxs[0].tolist() if 0 <= i < len(self.id_map)]

    def _ensure_clip(self):
        if self.clip_embedder is None:
            from clip_embedder import ClipEmbedder
            self.clip_embedder = ClipEmbedder.shared()
        return self.clip_embedder

    def _image_search(self, query: str, k: int) -> list[SourceResult]:
        if self.image_index is None or not self.image_id_map:
            return []
        clip = self._ensure_clip()
        vec = clip.embed_text_one(query).astype("float32").reshape(1, -1)
        scores, idxs = self.image_index.search(vec, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()
        att_ids = [self.image_id_map[i] for i in idxs if 0 <= i < len(self.image_id_map)]
        if not att_ids:
            return []
        placeholders = ",".join("?" * len(att_ids))
        with self._open_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM images WHERE attachment_id IN ({placeholders})", att_ids
            ).fetchall()
        by_id = {r["attachment_id"]: r for r in rows}
        out: list[SourceResult] = []
        for att_id, score in zip(att_ids, scores):
            r = by_id.get(att_id)
            if not r:
                continue
            sender = r["sender_name"] or "Unknown"
            chat_title = r["chat_title"]
            out.append(
                SourceResult(
                    source="image",
                    contact_names=[sender],
                    date_start=r["date_iso"] or "",
                    date_end=r["date_iso"] or "",
                    score=float(score),
                    messages=[],
                    subject=None,
                    chat_title=chat_title,
                    snippet=(r["msg_text"] or "")[:280],
                    image_url=f"/attachment/{att_id}",
                    image_caption=r["msg_text"] or None,
                    attachment_id=att_id,
                )
            )
        return out

    def get_image_path(self, attachment_id: int) -> str | None:
        with self._open_db() as conn:
            row = conn.execute(
                "SELECT path FROM images WHERE attachment_id = ?", (attachment_id,)
            ).fetchone()
        return row["path"] if row else None

    def _fts_search(self, query: str, k: int) -> list[str]:
        match = _build_fts_match(query)
        if not match:
            return []
        with self._open_db() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?",
                (match, k),
            ).fetchall()
        return [r["chunk_id"] for r in rows]

    def _date_search(self, after_iso: str | None, before_iso: str | None, k: int) -> list[str]:
        """Return chunks whose date_start falls in the window, ordered by recency.
        Acts as a third retriever when a temporal phrase narrows the search."""
        if not after_iso and not before_iso:
            return []
        with self._open_db() as conn:
            params: list = []
            clauses: list[str] = []
            if after_iso:
                clauses.append("date_start >= ?")
                params.append(after_iso)
            if before_iso:
                clauses.append("date_end <= ?")
                params.append(before_iso)
            params.append(k)
            sql = (
                "SELECT chunk_id FROM chunks WHERE " + " AND ".join(clauses) +
                " ORDER BY date_start DESC LIMIT ?"
            )
            rows = conn.execute(sql, params).fetchall()
        return [r["chunk_id"] for r in rows]

    @staticmethod
    def _rrf(*ranked_lists: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranking in ranked_lists:
            for rank, chunk_id in enumerate(ranking):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _hydrate(
        self,
        ranked: list[tuple[str, float]],
        top_k: int,
        date_range: tuple[str | None, str | None] | None = None,
        query: str = "",
        source_filter: set[str] | None = None,
        contact_filter: set[str] | None = None,
    ) -> list[SourceResult]:
        """Hydrate fused chunk_ids into SourceResults. Filters (date, source,
        contact) drop rows post-fetch — we over-fetch to compensate so the
        final list still hits top_k when filters are active."""
        if not ranked:
            return []
        has_filter = bool(date_range) or source_filter is not None or contact_filter is not None
        fetch_n = min(len(ranked), top_k * 8 if has_filter else top_k)
        chunk_ids = [cid for cid, _ in ranked[:fetch_n]]
        placeholders = ",".join("?" * len(chunk_ids))
        with self._open_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
            ).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        out: list[SourceResult] = []
        after_iso, before_iso = date_range or (None, None)
        q_tokens = _query_tokens(query) if query else set()
        for chunk_id, score in ranked[:fetch_n]:
            r = by_id.get(chunk_id)
            if not r:
                continue
            if source_filter is not None and r["source"] not in source_filter:
                continue
            if after_iso and (r["date_start"] or "") < after_iso:
                continue
            if before_iso and (r["date_end"] or "") > before_iso:
                continue
            if contact_filter is not None:
                try:
                    chunk_contacts = set(json.loads(r["contact_names"] or "[]"))
                except (TypeError, ValueError):
                    chunk_contacts = set()
                if not (chunk_contacts & contact_filter):
                    continue
            messages_json = r["messages"] or "[]"
            messages = [ChunkMessage(**m) for m in json.loads(messages_json)]
            # Tag the message that best matches the query so the UI can anchor
            # the preview around it instead of always showing the chunk's tail.
            if messages and q_tokens:
                best = _best_message_index(messages, q_tokens)
                messages[best].is_best_match = True
            snippet = r["text"] or ""
            out.append(
                SourceResult(
                    source=r["source"],
                    contact_names=json.loads(r["contact_names"]),
                    date_start=r["date_start"],
                    date_end=r["date_end"],
                    score=float(score),
                    messages=messages,
                    subject=r["subject"],
                    chat_title=r["chat_title"],
                    snippet=snippet[:280] if not messages else "",
                )
            )
            if len(out) >= top_k:
                break
        return out

    # -- Public --

    async def search(self, req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        intent = req.intent or QueryIntent()
        if req.intent is not None:
            print(
                f"[search] query=\"{req.query}\" intent: topic=\"{intent.topic}\" "
                f"sources={intent.sources} contacts={intent.contacts} "
                f"attachment={intent.must_have_attachment}",
                file=sys.stderr,
            )

        # 1) Temporal stripping — independent of intent. We still want
        #    "yesterday" / "last week" detection on top of source filters.
        cleaned_query, trange = temporal.parse(req.query)
        # 2) The on-device router gives us the semantic core. Prefer it,
        #    fall back to the temporal-stripped query, then the raw query.
        topic = (intent.topic or "").strip()
        embed_query = topic or (cleaned_query if cleaned_query.strip() else req.query)
        date_filter = trange.to_iso_range() if trange else None

        # 3) Translate intent into retrieval-time filters.
        source_filter: set[str] | None = set(intent.sources) if intent.sources else None
        contact_filter: set[str] | None = None
        if intent.contacts:
            resolved = self._resolve_contacts(intent.contacts)
            if resolved:
                contact_filter = resolved
        # Fallback: the on-device LLM sometimes misses contact names that
        # are clearly in the query. Tokenize the raw query and pick up any
        # tokens that map to a known canonical contact name.
        fallback = self._extract_contacts_from_query(req.query)
        if fallback:
            if contact_filter is None:
                contact_filter = fallback
            else:
                contact_filter = contact_filter | fallback
            if not intent.contacts or not self._resolve_contacts(intent.contacts):
                print(
                    f"[search] fallback contact extraction added: {sorted(fallback)}",
                    file=sys.stderr,
                )
        # If the user explicitly asked for an image, treat image as the only
        # relevant source — text excerpts about a photo aren't what they want.
        if intent.must_have_attachment:
            source_filter = {"image"}

        # Decide which retrievers to launch. Skip image if the user excluded
        # it; skip text retrievers when image is the only allowed source.
        run_text = source_filter is None or bool(source_filter - {"image"})
        run_image = source_filter is None or "image" in source_filter

        dense_task = asyncio.create_task(asyncio.to_thread(self._dense_search, embed_query, FUSE_FETCH * (4 if source_filter else 1))) if run_text else None
        fts_task = asyncio.create_task(asyncio.to_thread(self._fts_search, embed_query, FUSE_FETCH * (4 if source_filter else 1))) if run_text else None
        image_task = asyncio.create_task(asyncio.to_thread(self._image_search, embed_query, max(4, req.top_k))) if run_image else None
        remote_task = asyncio.create_task(hyperspell.search(req.query, top_k=req.top_k))
        date_task = None
        if date_filter and run_text:
            after_iso, before_iso = date_filter
            date_task = asyncio.create_task(
                asyncio.to_thread(self._date_search, after_iso, before_iso, FUSE_FETCH)
            )

        tasks: list = [t for t in (dense_task, fts_task, image_task, remote_task, date_task) if t is not None]
        results_list = await asyncio.gather(*tasks)
        results = dict(zip([t for t in (dense_task, fts_task, image_task, remote_task, date_task) if t is not None], results_list))
        dense_ids = results.get(dense_task, [])
        fts_ids = results.get(fts_task, [])
        image_results = results.get(image_task, [])
        remote_results = results.get(remote_task, [])
        date_ids = results.get(date_task, [])

        if date_ids:
            fused = self._rrf(dense_ids, fts_ids, date_ids)
        else:
            fused = self._rrf(dense_ids, fts_ids)

        local = self._hydrate(
            fused,
            req.top_k,
            date_range=date_filter,
            query=embed_query,
            source_filter=source_filter,
            contact_filter=contact_filter,
        )
        if date_filter:
            after_iso, before_iso = date_filter
            image_results = [
                r for r in image_results
                if (not after_iso or (r.date_start or "") >= after_iso)
                and (not before_iso or (r.date_end or "") <= before_iso)
            ]
        if contact_filter is not None:
            image_results = [
                r for r in image_results
                if any(name in contact_filter for name in (r.contact_names or []))
            ]

        # Fallback: if hard filters wiped everything out, retry without
        # contact/source filters so the user still sees results rather than
        # a confused empty state. We log so misroutes are visible.
        if not local and not image_results and (source_filter or contact_filter):
            print(
                f"[search] filtered to empty (sources={source_filter}, contacts={contact_filter}); "
                f"retrying unfiltered",
                file=sys.stderr,
            )
            local = self._hydrate(fused, req.top_k, date_range=date_filter, query=embed_query)

        merged = self._merge_results(local, image_results, remote_results, max_total=req.top_k)
        answer = await self._synthesize(req.query, merged, trange, target_contacts=contact_filter)
        return SearchResponse(
            answer=answer,
            sources=merged,
            query_ms=int((time.perf_counter() - t0) * 1000),
        )

    @staticmethod
    def _merge_results(
        local: list[SourceResult],
        images: list[SourceResult],
        remote: list[SourceResult],
        max_total: int,
    ) -> list[SourceResult]:
        # Interleave text + image roughly 2:1 so images get represented but don't
        # dominate. Then append any remote leftovers.
        out: list[SourceResult] = []
        ti, ii = 0, 0
        while len(out) < max_total and (ti < len(local) or ii < len(images)):
            for _ in range(2):
                if ti < len(local) and len(out) < max_total:
                    out.append(local[ti]); ti += 1
            if ii < len(images) and len(out) < max_total:
                out.append(images[ii]); ii += 1
        for r in remote:
            if len(out) >= max_total:
                break
            out.append(r)
        return out[:max_total]


    async def _synthesize(
        self,
        query: str,
        sources: list[SourceResult],
        trange: temporal.TemporalRange | None = None,
        target_contacts: set[str] | None = None,
    ) -> str:
        """Synthesize an answer from retrieved excerpts.

        When `target_contacts` is non-empty, we route to a separate path that
        feeds the LLM ONLY the contact's own lines and uses a focused prompt.
        Past attempts kept the user's words in via labels and lost — the LLM
        consistently leaked user opinions back as the contact's. The simpler
        the input, the safer the output.
        """
        if not sources or not os.getenv("OPENAI_API_KEY"):
            return ""
        if not _looks_like_question(query):
            return ""

        if target_contacts:
            return await self._synthesize_contact_focused(
                query, sources, trange, target_contacts
            )

        excerpts: list[str] = []
        for s in sources:
            who = ", ".join(s.contact_names) or "Unknown"
            date = (s.date_start or "")[:10]
            if s.messages:
                lines: list[str] = []
                for m in s.messages:
                    tag = "" if m.known or m.is_from_me else " (new contact, not in address book)"
                    lines.append(f"{m.sender}{tag}: {m.text}")
                full_text = "\n".join(lines)
            else:
                full_text = s.snippet
            excerpts.append(
                f"[SOURCE: {_label(s.source)} | CONTACT: {who} | DATE: {date}]\n{full_text}\n---"
            )
        context = "\n".join(excerpts)

        # Inject relationship summaries for any contacts that appear in the
        # excerpts. This grounds the LLM in who the user actually knows
        # (cluster-sampled summaries built at index time).
        contact_block = ""
        if self._contact_summaries:
            unique_contacts: list[str] = []
            for s in sources:
                for name in s.contact_names:
                    if name in self._contact_summaries and name not in unique_contacts:
                        unique_contacts.append(name)
            if unique_contacts:
                lines = []
                for name in unique_contacts[:6]:
                    cs = self._contact_summaries[name]
                    lines.append(
                        f"- {name}: {cs['summary']}  "
                        f"(known since {cs['first']}, {cs['total_chunks']} chunks total, "
                        f"{cs['last_30d_chunks']} in last 30 days)"
                    )
                contact_block = "Known contacts in this excerpt set:\n" + "\n".join(lines) + "\n\n"
        # Pass current date and any detected temporal range to the LLM so it can
        # reason about "yesterday" / "last week" against the excerpt dates.
        import datetime as _dt
        today = _dt.date.today().isoformat()
        date_note = f"Today is {today}."

        if trange:
            after = trange.after.date().isoformat() if trange.after else "?"
            before = trange.before.date().isoformat() if trange.before else "?"
            date_note += (
                f" The user's query refers to '{trange.label}' which means dates between "
                f"{after} and {before}. Only use excerpts in that range; ignore any older "
                "matches even if they contain similar words."
            )
        try:
            resp = await self._openai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{date_note}\n\n{contact_block}Query: {query}\n\nExcerpts:\n{context}"},
                ],
            )
            answer = (resp.choices[0].message.content or "").strip()
            # Defensive: if the model ignored instructions and returned a non-answer
            # placeholder, normalize back to an actual empty string.
            if _looks_like_non_answer(answer):
                return ""
            return answer
        except Exception:
            return ""

    async def _synthesize_contact_focused(
        self,
        query: str,
        sources: list[SourceResult],
        trange: temporal.TemporalRange | None,
        target_contacts: set[str],
    ) -> str:
        """Strict path: the LLM sees ONLY the targeted contacts' own messages.

        No user lines, no other speakers, no [EVIDENCE]/[CONTEXT] labels —
        the format is unmistakable: every line is a message the contact
        sent. Pattern/tone inference is still encouraged by the prompt.
        """
        # Gather only the contact's lines across all sources, with light
        # source headers so the model knows messages come from different
        # threads/dates.
        blocks: list[str] = []
        total_lines = 0
        for s in sources:
            if not s.messages:
                continue
            contact_lines = _filter_for_target_contacts(s.messages, target_contacts)
            if not contact_lines:
                continue
            date = (s.date_start or "")[:10]
            chat = s.chat_title or ""
            header_bits = [_label(s.source), date]
            if chat:
                header_bits.append(chat)
            header = " · ".join(b for b in header_bits if b)
            body = "\n".join(f"{m.sender}: {m.text}" for m in contact_lines)
            blocks.append(f"[{header}]\n{body}")
            total_lines += len(contact_lines)
        if total_lines == 0:
            # No contact-spoken lines in any excerpt — nothing to synthesize.
            return ""

        names = ", ".join(sorted(target_contacts))
        context = "\n\n".join(blocks)

        # Deliberately omit contact_summary here. The summary is the only
        # other place where user-phrased content could reach the LLM in
        # this path; keeping it out makes the input strictly the contact's
        # own words, which is what the quote validator also checks against.

        import datetime as _dt
        today = _dt.date.today().isoformat()
        date_note = f"Today is {today}."
        if trange:
            after = trange.after.date().isoformat() if trange.after else "?"
            before = trange.before.date().isoformat() if trange.before else "?"
            date_note += (
                f" The query refers to '{trange.label}' (between {after} and {before}). "
                f"Only use excerpts in that range."
            )

        user_content = (
            f"{date_note}\n\n"
            f"Query: {query}\n\n"
            f"Messages from {names} (the only person/people whose words you can see):\n"
            f"{context}"
        )
        try:
            resp = await self._openai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": CONTACT_FOCUSED_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            answer = (resp.choices[0].message.content or "").strip()
            if _looks_like_non_answer(answer):
                return ""
            # Strict guard against fabricated quotes. If the LLM puts text in
            # quotation marks, that text MUST appear in the messages we sent
            # — otherwise it's hallucinated and the whole answer is rejected.
            ok, bad = _validate_no_invented_quotes(answer, context)
            if not ok:
                print(
                    f"[synthesis] rejecting answer due to invented quote: "
                    f"{bad!r}; returning empty",
                    file=sys.stderr,
                )
                return ""
            return answer
        except Exception:
            return ""


_QUESTION_LEAD_WORDS = {
    "what", "who", "where", "when", "why", "how", "does", "do", "did",
    "is", "are", "was", "were", "has", "have", "had", "can", "could",
    "should", "will", "would", "tell", "explain", "summarize", "list",
    "show", "find", "any", "which",
}


def _looks_like_question(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    if q.endswith("?"):
        return True
    first = q.split(maxsplit=1)[0] if q else ""
    # Strip contractions: "what's" → "what", "who're" → "who", etc.
    bare = re.sub(r"['’](?:s|re|ve|d|ll|m)$", "", first)
    if first in _QUESTION_LEAD_WORDS or bare in _QUESTION_LEAD_WORDS:
        return True
    return False


_NON_ANSWER_PATTERNS = (
    "empty",
    "no answer",
    "no relevant",
    "i cannot",
    "i can't",
    "i don't have",
    "(nothing)",
    "n/a",
)


# Only double-quote pairs count as a quotation for validation. Single quotes
# / apostrophes are too ambiguous in English (possessives, contractions) to
# treat as quote delimiters without false positives.
_QUOTE_RE = re.compile(r'["“”]([^"“”\n]{2,120})["“”]')
_NORMALIZE_QUOTE_RE = re.compile(r"\s+")


def _normalize_for_compare(s: str) -> str:
    # Lowercase, collapse whitespace, strip trailing/leading punctuation
    # that the LLM might add when embedding a quote in its own sentence.
    cleaned = _NORMALIZE_QUOTE_RE.sub(" ", s.lower()).strip()
    return cleaned.strip(".,;:!?…\"'“”‘’ ")


def _validate_no_invented_quotes(answer: str, allowed_text: str) -> tuple[bool, str | None]:
    """Every quoted phrase in `answer` must appear (case-insensitively, with
    whitespace collapsed, and ignoring surrounding punctuation) inside
    `allowed_text`. Returns (ok, offending_quote).

    Defense against gpt-4o-mini fabricating "alarm said 'so peak'" when those
    words were never in the excerpts. A single fabricated quote rejects the
    whole answer — the user gets nothing instead of misinformation.
    """
    haystack = _normalize_for_compare(allowed_text)
    for m in _QUOTE_RE.finditer(answer):
        quoted = m.group(1).strip()
        if len(quoted) < 3:
            continue
        norm = _normalize_for_compare(quoted)
        if not norm:
            continue
        if norm not in haystack:
            return False, quoted
    return True, None


def _looks_like_non_answer(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower().strip(".\"'() ")
    if len(t) < 30 and any(p in t for p in _NON_ANSWER_PATTERNS):
        return True
    return False


def _label(source: str) -> str:
    return {"imessage": "iMessage", "mail": "Mail", "hyperspell": "Hyperspell"}.get(source, source.title())


_FTS_TOKEN_SAFE = re.compile(r"[A-Za-z0-9]+")


def _build_fts_match(query: str) -> str:
    """Tokenize a free-text query into an FTS5 MATCH expression with prefix tokens.

    Uses OR so we maximize recall (a chunk mentioning *either* term is a candidate);
    RRF will surface the strongest matches when fused with dense retrieval.
    Stripped of stop-words and short tokens to avoid noise.
    """
    tokens = [t.lower() for t in _FTS_TOKEN_SAFE.findall(query) if len(t) >= 2]
    tokens = [t for t in tokens if t not in _STOP]
    if not tokens:
        return ""
    return " OR ".join(f"{t}*" for t in tokens)


_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "but", "if", "then", "this", "that", "it",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "they",
    "what", "who", "when", "where", "why", "how", "any", "some", "all",
    "about", "from", "by", "as",
}
