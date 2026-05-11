"""Orchestrate parsing → chunking → embedding → FAISS index + metadata.db."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from chunker import Chunk, chunk_imessages, chunk_mail_threads
from contacts import ContactResolver
from embedder import EMBED_DIM, Embedder
from parse_imessage import iter_messages

DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
META_DB_PATH = DATA_DIR / "metadata.db"
ID_MAP_PATH = DATA_DIR / "id_map.json"
IMAGE_INDEX_PATH = DATA_DIR / "images.faiss"
IMAGE_ID_MAP_PATH = DATA_DIR / "image_id_map.json"

IVF_THRESHOLD = 100_000
IVF_NLIST = 256


def _init_metadata_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            source TEXT,
            contact_names TEXT,
            date_start TEXT,
            date_end TEXT,
            text TEXT,
            row_ids TEXT,
            messages TEXT,
            subject TEXT,
            chat_title TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            text,
            tokenize = "unicode61 remove_diacritics 2"
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            attachment_id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            mime TEXT,
            date_iso TEXT,
            sender_name TEXT,
            sender_known INTEGER,
            contact_key TEXT,
            chat_title TEXT,
            chat_id INTEGER,
            is_from_me INTEGER,
            msg_text TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_date ON images(date_iso)")
    return conn


def _persist_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    rows = [
        (
            c.chunk_id,
            c.source,
            json.dumps(c.contact_names),
            c.date_start,
            c.date_end,
            c.text,
            json.dumps(c.row_ids),
            json.dumps([asdict(m) for m in c.messages]),
            c.subject,
            c.chat_title,
        )
        for c in chunks
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks
        (chunk_id, source, contact_names, date_start, date_end, text, row_ids, messages, subject, chat_title)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.executemany(
        "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
        [(c.chunk_id, c.text) for c in chunks],
    )
    conn.commit()


def _build_faiss(vectors: np.ndarray, dim: int | None = None) -> faiss.Index:
    n = vectors.shape[0]
    d = dim if dim is not None else EMBED_DIM
    if n < IVF_THRESHOLD:
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
        return index
    quantizer = faiss.IndexFlatIP(d)
    nlist = min(IVF_NLIST, max(1, n // 40))
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = max(8, nlist // 16)
    return index


def _collect_chunks(
    sources: set[str],
    resolver: ContactResolver,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    from me_identity import discover as discover_me
    user_emails, _ = discover_me()
    print(f"  user identified by {len(user_emails)} email(s)", file=sys.stderr)
    chunks: list[Chunk] = []
    if "imessage" in sources:
        print("Parsing iMessage…", file=sys.stderr)
        rows = list(iter_messages())
        print(f"  {len(rows):,} messages", file=sys.stderr)
        # Pass the embedder through so chunk_imessages can use it for the
        # ambient-bucket cohesion segmentation.
        ic = list(chunk_imessages(rows, resolver=resolver, embedder=embedder))
        print(f"  → {len(ic):,} chunks", file=sys.stderr)
        chunks.extend(ic)
    if "mail" in sources:
        print("Parsing Apple Mail…", file=sys.stderr)
        try:
            from parse_mail import iter_mail_threads
            threads = list(iter_mail_threads())
            print(f"  {len(threads):,} threads", file=sys.stderr)
            mc = list(chunk_mail_threads(threads, resolver=resolver, user_emails=user_emails))
            print(f"  → {len(mc):,} chunks", file=sys.stderr)
            chunks.extend(mc)
        except FileNotFoundError as e:
            print(f"  skipping mail: {e}", file=sys.stderr)
    return chunks


def _build_image_index(resolver: ContactResolver, conn: sqlite3.Connection) -> int:
    """Walk attachment table, embed each image with CLIP, save FAISS + metadata.

    Returns the number of images successfully indexed.
    """
    from parse_attachments import iter_image_attachments
    from clip_embedder import CLIP_DIM, ClipEmbedder

    print("Enumerating image attachments…", file=sys.stderr)
    attachments = list(iter_image_attachments(resolver=resolver))
    print(f"  {len(attachments):,} image files on disk", file=sys.stderr)
    if not attachments:
        return 0

    embedder = ClipEmbedder.shared()
    print(f"Embedding images with CLIP (this is the slow part)…", file=sys.stderr)
    BATCH = 32
    all_vecs: list[np.ndarray] = []
    kept_atts: list = []
    for i in tqdm(range(0, len(attachments), BATCH), desc="image batches"):
        batch = attachments[i : i + BATCH]
        paths = [a.path for a in batch]
        vecs, kept_idx = embedder.embed_images(paths, batch_size=BATCH)
        if vecs.shape[0] == 0:
            continue
        all_vecs.append(vecs)
        for idx in kept_idx:
            kept_atts.append(batch[idx])

    if not all_vecs:
        return 0

    matrix = np.vstack(all_vecs)
    print(f"  built {matrix.shape[0]:,} image vectors", file=sys.stderr)

    index = _build_faiss(matrix, dim=CLIP_DIM)
    faiss.write_index(index, str(IMAGE_INDEX_PATH))
    IMAGE_ID_MAP_PATH.write_text(json.dumps([a.attachment_id for a in kept_atts]))

    rows = [
        (
            a.attachment_id,
            str(a.path),
            a.mime,
            a.date_iso,
            a.sender_name,
            int(a.sender_known),
            a.contact_key,
            a.chat_title,
            a.chat_id,
            int(a.is_from_me),
            a.msg_text,
        )
        for a in kept_atts
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO images "
        "(attachment_id, path, mime, date_iso, sender_name, sender_known, contact_key, "
        " chat_title, chat_id, is_from_me, msg_text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return matrix.shape[0]


def build(sources: set[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Contacts…", file=sys.stderr)
    resolver = ContactResolver()
    resolver.load()
    print(f"  {len(resolver._by_phone):,} phones, {len(resolver._by_email):,} emails", file=sys.stderr)
    n_photos = resolver.export_photos()
    print(f"  exported {n_photos:,} photos", file=sys.stderr)

    do_text = bool(sources - {"images"})
    do_images = "images" in sources
    n_images = 0

    if do_text:
        # Construct the embedder up front so it can be reused by both the
        # chunker (for ambient-bucket cohesion segmentation) AND the chunk-
        # text embedding pass below — avoids loading the model twice.
        print(f"Loading embedder (all-MiniLM-L6-v2)…", file=sys.stderr)
        embedder = Embedder()
        chunks = _collect_chunks(sources - {"images"}, resolver, embedder=embedder)
        if not chunks:
            print("No text chunks produced.", file=sys.stderr)
        else:
            print(f"Embedding {len(chunks):,} chunks…", file=sys.stderr)
            texts = [c.text for c in chunks]
            vectors = embedder.embed_batch(texts).astype("float32")
            print("Building text FAISS index…", file=sys.stderr)
            index = _build_faiss(vectors)
            faiss.write_index(index, str(INDEX_PATH))
            print("Writing metadata.db…", file=sys.stderr)
            # Preserve the (slow-to-rebuild) `images` table across text rebuilds.
            # On a near-full disk we can't journal a DELETE/DROP, so we dump
            # images to memory, recreate the file fresh, then re-insert.
            preserved_images: list[tuple] = []
            if META_DB_PATH.exists():
                try:
                    src = sqlite3.connect(META_DB_PATH)
                    src.execute("PRAGMA journal_mode = OFF")
                    cur = src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
                    if cur.fetchone():
                        preserved_images = list(src.execute("SELECT * FROM images"))
                        print(f"  preserving {len(preserved_images):,} image rows…", file=sys.stderr)
                    src.close()
                except sqlite3.Error as e:
                    print(f"  could not preserve images: {e}", file=sys.stderr)
                META_DB_PATH.unlink()
            conn = _init_metadata_db(META_DB_PATH)
            if preserved_images:
                conn.executemany(
                    "INSERT INTO images "
                    "(attachment_id, path, mime, date_iso, sender_name, sender_known, "
                    " contact_key, chat_title, chat_id, is_from_me, msg_text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    preserved_images,
                )
                conn.commit()
            BATCH = 500
            for i in tqdm(range(0, len(chunks), BATCH), desc="metadata"):
                _persist_chunks(conn, chunks[i : i + BATCH])
            ID_MAP_PATH.write_text(json.dumps([c.chunk_id for c in chunks]))

            if do_images:
                n_images = _build_image_index(resolver, conn)

            conn.close()

            counts: dict[str, int] = {}
            for c in chunks:
                counts[c.source] = counts.get(c.source, 0) + 1
            print("\n=== Index built ===", file=sys.stderr)
            for s, n in counts.items():
                print(f"  {s:<10} {n:,} chunks", file=sys.stderr)
            if n_images:
                print(f"  images    {n_images:,} embedded", file=sys.stderr)
            print(f"  TOTAL    {len(chunks):,} text chunks + {n_images:,} images", file=sys.stderr)
            print(f"  text idx  {INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB", file=sys.stderr)
            if IMAGE_INDEX_PATH.exists():
                print(f"  image idx {IMAGE_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB", file=sys.stderr)
            print(f"  metadata  {META_DB_PATH.stat().st_size / 1024 / 1024:.1f} MB", file=sys.stderr)
    elif do_images:
        # Image-only build: keep existing text index & metadata, just (re)build images.
        if not META_DB_PATH.exists():
            print("metadata.db missing — run a text build first.", file=sys.stderr)
            return
        conn = _init_metadata_db(META_DB_PATH)
        n_images = _build_image_index(resolver, conn)
        conn.close()
        print(f"\n=== Image index built ===\n  {n_images:,} images embedded", file=sys.stderr)


def _build_contact_summaries() -> None:
    from contact_summaries import build as build_cs
    build_cs(META_DB_PATH, INDEX_PATH, ID_MAP_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["imessage"],
        choices=["imessage", "mail", "images"],
        help="Sources to include in this build (images = CLIP image attachments)",
    )
    parser.add_argument(
        "--summaries",
        action="store_true",
        help="Build/refresh per-contact relationship summaries (LLM, ~1¢)",
    )
    args = parser.parse_args()
    build(set(args.sources))
    if args.summaries:
        _build_contact_summaries()


if __name__ == "__main__":
    main()
