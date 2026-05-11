"""Per-contact relationship summaries via embedding-space cluster sampling.

The naive approach — sample N most-recent messages → LLM summary — fails because
recent messages are biased toward whatever's current (random links, memes,
"on my way"). The structure of the embedding space encodes the actual topical
breadth of a relationship: messages about dating cluster together, messages
about school cluster together, etc. We harvest that structure directly.

Algorithm per contact:
  1. Find every chunk this contact appears in (via the contacts JSON column).
  2. Reconstruct those chunks' vectors from the FAISS index.
  3. K-means cluster into ~min(8, n_chunks/3) clusters.
  4. For each cluster, find the chunk closest to the centroid — that's the most
     representative chunk for that topic blob.
  5. Sample 1 substantive message from each representative chunk.
  6. Send all samples + stats (first contact, total exchanged, recent activity)
     to gpt-4o-mini → one factual relationship sentence.
  7. Cache by hash so unchanged contacts skip the LLM on re-runs.

Total cost across ~50 contacts is < 1¢. Refresh weekly — the structure is
stable on that timescale even with active conversations.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sklearn.cluster import KMeans

from contacts import Contact, ContactResolver

OPENAI_MODEL = "gpt-4o-mini"
MAX_CLUSTERS = 8
MIN_CHUNKS_FOR_CLUSTERING = 6      # below this, just sample a few messages directly
MIN_CHUNKS_FOR_SUMMARY = 2

PROMPT = (
    "You receive a sample of representative messages between the user (labeled "
    "'Me') and one contact, drawn from across their conversation history. "
    "Output ONE factual sentence describing the user's relationship with this "
    "person: how they're connected (friend, parent, advisor, classmate, etc.), "
    "what they primarily talk about (recurring topics across the sample), and "
    "the apparent intimacy/activity level. "
    "No preamble. No hedging like 'it seems' or 'appears to'. No quotation "
    "marks around the whole answer. Maximum 30 words. "
    "If the sample is too thin to tell, output exactly: 'Unclear from sample.'"
)


@dataclass
class ContactSummary:
    name: str
    first_message_iso: str
    last_message_iso: str
    total_chunks: int          # chunks they appear in (rough activity proxy)
    last_30d_chunks: int       # chunks in last 30 days
    n_clusters_sampled: int    # how many topic clusters we found
    summary: str


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS contact_summaries (
            name TEXT PRIMARY KEY,
            first_message_iso TEXT,
            last_message_iso TEXT,
            total_chunks INTEGER,
            last_30d_chunks INTEGER,
            n_clusters_sampled INTEGER,
            summary TEXT,
            sample_hash TEXT
        )
        """
    )
    return conn


def _existing_hash(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute("SELECT sample_hash FROM contact_summaries WHERE name=?", (name,)).fetchone()
    return row[0] if row else None


def _save(conn: sqlite3.Connection, s: ContactSummary, sample_hash: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO contact_summaries "
        "(name, first_message_iso, last_message_iso, total_chunks, last_30d_chunks, "
        " n_clusters_sampled, summary, sample_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (s.name, s.first_message_iso, s.last_message_iso, s.total_chunks,
         s.last_30d_chunks, s.n_clusters_sampled, s.summary, sample_hash),
    )
    conn.commit()


def _chunks_for_contact(conn: sqlite3.Connection, name: str, id_map_pos: dict[str, int]) -> list[tuple[int, dict, str, str]]:
    """Return [(faiss_pos, messages, date_start, chunk_id), ...] for one contact.

    Matches via JSON contains on contact_names (case-sensitive — names are
    normalized at index time so this is safe).
    """
    quoted = json.dumps(name)              # → "Brendan Giang"
    like = f'%{quoted[1:-1]}%'             # crude but effective; safer than json_each
    rows = conn.execute(
        "SELECT chunk_id, messages, date_start FROM chunks "
        "WHERE source='imessage' AND contact_names LIKE ?",
        (like,),
    ).fetchall()
    out: list[tuple[int, dict, str, str]] = []
    for chunk_id, messages_json, date_start in rows:
        pos = id_map_pos.get(chunk_id)
        if pos is None:
            continue
        try:
            msgs = json.loads(messages_json or "[]")
        except json.JSONDecodeError:
            continue
        out.append((pos, msgs, date_start or "", chunk_id))
    return out


def _representative_message(messages: list[dict]) -> str | None:
    """Pick the longest non-trivial message in the chunk."""
    candidates = [m["text"].strip() for m in messages if m.get("text") and len(m["text"].strip()) > 12]
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    text = candidates[0]
    return text[:240]


def _cluster_sample(
    index: faiss.Index,
    chunks: list[tuple[int, dict, str, str]],
    n_clusters: int,
) -> tuple[list[str], int]:
    """Return (representative_messages, actual_n_clusters)."""
    n = len(chunks)
    if n < MIN_CHUNKS_FOR_CLUSTERING:
        # Too few for clustering — sample messages directly.
        samples: list[str] = []
        for _, msgs, _, _ in chunks:
            m = _representative_message(msgs)
            if m:
                samples.append(m)
        return samples[:n_clusters], min(n, n_clusters)

    vectors = np.vstack([index.reconstruct(pos).reshape(1, -1) for pos, *_ in chunks])
    k = min(n_clusters, max(2, n // 3))
    km = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = km.fit_predict(vectors)
    samples: list[str] = []
    for cluster_id in range(k):
        in_cluster = np.where(labels == cluster_id)[0]
        if len(in_cluster) == 0:
            continue
        # Find chunk closest to this cluster's centroid.
        centroid = km.cluster_centers_[cluster_id]
        cluster_vecs = vectors[in_cluster]
        # Vectors are normalized → cosine = dot.
        sims = cluster_vecs @ centroid / (np.linalg.norm(centroid) or 1.0)
        best_local = int(np.argmax(sims))
        best_chunk = chunks[in_cluster[best_local]]
        m = _representative_message(best_chunk[1])
        if m:
            samples.append(m)
    return samples, k


def _stats(chunks: list[tuple[int, dict, str, str]]) -> tuple[str, str, int, int]:
    if not chunks:
        return ("", "", 0, 0)
    dates = sorted(c[2] for c in chunks if c[2])
    first = dates[0] if dates else ""
    last = dates[-1] if dates else ""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat()
    last_30 = sum(1 for c in chunks if c[2] and c[2] >= cutoff)
    return first, last, len(chunks), last_30


def _hash(samples: list[str], name: str) -> str:
    payload = name + "\n" + "\n---\n".join(samples)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build(metadata_db: Path, index_path: Path, id_map_path: Path,
          openai_api_key: str | None = None) -> int:
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("contact_summaries: OPENAI_API_KEY not set; skipping LLM summarization", file=sys.stderr)
        return 0
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print("Loading FAISS + id_map…", file=sys.stderr)
    index = faiss.read_index(str(index_path))
    id_list = json.loads(id_map_path.read_text())
    id_map_pos = {cid: i for i, cid in enumerate(id_list)}

    print("Loading Contacts…", file=sys.stderr)
    resolver = ContactResolver()
    resolver.load()
    # Dedupe contacts by name; merge handles per person.
    contacts_by_name: dict[str, Contact] = {}
    for c in (*resolver._by_phone.values(), *resolver._by_email.values()):
        if c.name not in contacts_by_name:
            contacts_by_name[c.name] = c
    print(f"  {len(contacts_by_name)} unique contacts", file=sys.stderr)

    out_conn = init_db(metadata_db)
    src_conn = sqlite3.connect(metadata_db)

    written, skipped, light = 0, 0, 0
    for name in sorted(contacts_by_name):
        chunks = _chunks_for_contact(src_conn, name, id_map_pos)
        if len(chunks) < MIN_CHUNKS_FOR_SUMMARY:
            continue
        samples, k = _cluster_sample(index, chunks, MAX_CLUSTERS)
        if not samples:
            continue
        if k < 3:
            light += 1
        first, last, total, recent = _stats(chunks)
        sample_hash = _hash(samples, name)
        if _existing_hash(out_conn, name) == sample_hash:
            skipped += 1
            continue

        sample_block = "\n\n".join(f"• {s}" for s in samples)
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                max_tokens=80,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": (
                        f"Contact: {name}\n"
                        f"Total chunks (sliding 20-msg windows): {total}\n"
                        f"Last 30 days: {recent} chunks\n"
                        f"First contact: {first[:10] if first else '?'}\n"
                        f"Most-recent contact: {last[:10] if last else '?'}\n"
                        f"Cluster-sampled representative messages "
                        f"({k} cluster{'s' if k != 1 else ''}):\n\n{sample_block}"
                    )},
                ],
            )
            summary_text = (resp.choices[0].message.content or "").strip().strip('"\'')
        except Exception as e:
            print(f"  {name}: summarize failed ({e})", file=sys.stderr)
            continue

        cs = ContactSummary(
            name=name,
            first_message_iso=first,
            last_message_iso=last,
            total_chunks=total,
            last_30d_chunks=recent,
            n_clusters_sampled=k,
            summary=summary_text,
        )
        _save(out_conn, cs, sample_hash)
        written += 1
        print(f"  [{k:2d}c, {total:>4d}ch, {recent:>3d} 30d] {name}: {summary_text}", file=sys.stderr)

    print(f"\ncontact_summaries: wrote {written}, cached {skipped}, "
          f"{light} contacts had too few chunks for full clustering", file=sys.stderr)
    return written


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / "api" / ".env")
    base = Path(__file__).parent / "data"
    n = build(base / "metadata.db", base / "index.faiss", base / "id_map.json")
    sys.exit(0 if n >= 0 else 1)
