#!/usr/bin/env python3
"""consolidate.py — CLI for the consolidation scan and its inspection tools.

  scan                 resume every thread from its checkpoint (small threads first)
  scan --thread KEY    scan one thread
  scan --fixture PATH  run a synthetic fixture through the real pipeline (throwaway db)
  scan --max-blocks N  bound a run (validation / dev-slice)
  show --entity NAME   aliases (with windows) + claims labeled CURRENT/EXPIRED/SUPERSEDED
  show --thread KEY    per-thread claims and contexts
  stats                store counts + drift indicators (new-entity rate, evidence-drop rate)
  merge                global entity merge (dry-run default; --apply to mutate; --unmerge ID)
  merge-review         list/inspect the unsure merge-adjudication queue
  dossiers             (re)generate entities.summary from top valid claims (claim-id cached)

  contextualize / nightly  -> later phases (stubs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import contextual_embed as ctx
import memory_merge as merge
import memory_scan as scan
import memory_store as store

METADATA_DB_PATH = store.DATA_DIR / "metadata.db"


def _excluded_thread_keys() -> set[str]:
    """Threads a default scan skips — SEMSE_EXCLUDED_THREADS, comma-separated.

    Read on call, not at import: the value usually arrives from api/.env, which
    _load_openai_env() only loads once a command is already running."""
    raw = os.environ.get("SEMSE_EXCLUDED_THREADS", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


def _load_openai_env() -> None:
    """OPENAI_API_KEY lives in api/.env — the same file the FastAPI app loads."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / "api" / ".env")
    except Exception:  # noqa: BLE001 — dotenv optional; env may already be set
        pass


# --------------------------------------------------------------------------- #
# scan                                                                         #
# --------------------------------------------------------------------------- #

def _thread_keys_by_size(metadata_db_path: Path, source: str = "imessage") -> list[tuple[str, int]]:
    src = sqlite3.connect(f"file:{metadata_db_path}?mode=ro", uri=True)
    try:
        counts: dict[str, int] = {}
        for tk, rid in src.execute(
            "SELECT thread_key, row_ids FROM chunks WHERE source=? AND thread_key IS NOT NULL",
            (source,)):
            counts[tk] = counts.get(tk, 0) + len(json.loads(rid or "[]"))
    finally:
        src.close()
    return sorted(counts.items(), key=lambda kv: kv[1])


def cmd_scan(args: argparse.Namespace) -> None:
    if args.fixture:
        _scan_fixture(args)
        return
    memory_db = Path(args.memory_db) if args.memory_db else store.MEMORY_DB_PATH
    conn = store.connect(memory_db)
    try:
        if args.thread:
            threads = [args.thread]
        else:
            threads = [tk for tk, _ in _thread_keys_by_size(METADATA_DB_PATH)]
            if not args.include_all_threads:
                excluded = _excluded_thread_keys()
                threads = [t for t in threads if t not in excluded]
        failures = 0
        for tk in threads:
            print(f"scanning {tk} ...", flush=True)
            try:
                summary = scan.scan_thread(conn, tk, metadata_db_path=METADATA_DB_PATH,
                                           max_blocks=args.max_blocks)
            except Exception as e:
                # A long unattended run must outlive one thread's failure; the failed
                # thread resumes from its last committed block on the next scan.
                failures += 1
                print(f"  !! {tk} failed ({type(e).__name__}: {e}) — continuing", flush=True)
                continue
            print(f"  -> blocks={summary['blocks']} entities={summary.get('entities',0)} "
                  f"claims={summary.get('claims',0)} mentions={summary.get('mentions',0)} "
                  f"ev_dropped={summary.get('evidence_dropped',0)} "
                  f"claims_dropped={summary.get('claims_dropped',0)} "
                  f"review_queued={summary.get('review_queued',0)}", flush=True)
        if failures:
            print(f"done with {failures} failed thread(s) — re-run scan to retry them",
                  flush=True)
    finally:
        conn.close()


def _scan_fixture(args: argparse.Namespace) -> None:
    memory_db = Path(args.memory_db) if args.memory_db else \
        store.DATA_DIR / "memory_fixture_scratch.db"
    if memory_db.exists():
        memory_db.unlink()  # throwaway — start clean
    paths = [Path(args.fixture)] if not Path(args.fixture).is_dir() else \
        sorted(Path(args.fixture).glob("*.json"))
    all_pass = True
    print(f"{'FIXTURE':<20} {'ARC':<16} {'RESULT':<6} DETAIL")
    for p in paths:
        for r in scan.run_fixture(p, memory_db):
            status = "PASS" if r["passed"] else "FAIL"
            all_pass = all_pass and r["passed"]
            fails = [name for name, ok in r["checks"] if not ok]
            detail = f"claims={r['n_claims']}" + (f" | failed: {', '.join(fails)}" if fails else "")
            print(f"{r['fixture']:<20} {r['arc']:<16} {status:<6} {detail}")
    print("\nALL FIXTURES PASSED" if all_pass else "\nSOME FIXTURES FAILED")
    sys.exit(0 if all_pass else 1)


# --------------------------------------------------------------------------- #
# show                                                                         #
# --------------------------------------------------------------------------- #

def _claim_label(row: sqlite3.Row, today: str) -> str:
    if row["superseded_by"] or row["invalidated_at"]:
        return "SUPERSEDED"
    if row["valid_until"] and row["valid_until"][:10] < today:
        return "EXPIRED"
    return "CURRENT"


def cmd_show(args: argparse.Namespace) -> None:
    conn = store.connect(Path(args.memory_db) if args.memory_db else store.MEMORY_DB_PATH)
    today = date.today().isoformat()
    try:
        if args.entity:
            _show_entity(conn, args.entity, today)
        elif args.thread:
            _show_thread(conn, args.thread, today)
        else:
            print("specify --entity NAME|ID or --thread KEY")
    finally:
        conn.close()


def _resolve_entity_arg(conn: sqlite3.Connection, arg: str) -> str | None:
    if conn.execute("SELECT 1 FROM entities WHERE entity_id=?", (arg,)).fetchone():
        return arg
    row = conn.execute(
        "SELECT entity_id FROM entities WHERE lower(canonical_name)=? AND merged_into IS NULL",
        (arg.lower(),)).fetchone()
    if row:
        return row[0]
    row = conn.execute(
        "SELECT entity_id FROM entities WHERE lower(canonical_name) LIKE ? "
        "AND merged_into IS NULL LIMIT 1", (f"%{arg.lower()}%",)).fetchone()
    if row:
        return row[0]
    row = conn.execute("SELECT entity_id FROM aliases WHERE norm=? LIMIT 1",
                       (store.norm_surface(arg),)).fetchone()
    return row[0] if row else None


def _show_entity(conn: sqlite3.Connection, arg: str, today: str) -> None:
    eid = _resolve_entity_arg(conn, arg)
    if not eid:
        print(f"no entity matching {arg!r}")
        return
    ent = conn.execute("SELECT * FROM entities WHERE entity_id=?", (eid,)).fetchone()
    print(f"# {ent['canonical_name']}  [{ent['kind']}]  ({eid})")
    print(f"  seen {ent['first_seen']} .. {ent['last_seen']}"
          f"{'  contact_key='+ent['contact_key'] if ent['contact_key'] else ''}")
    def _name_of(entity_id: str | None) -> str | None:
        if not entity_id:
            return None
        r = conn.execute("SELECT canonical_name FROM entities WHERE entity_id=?",
                         (entity_id,)).fetchone()
        return r[0] if r else entity_id

    print("  aliases:")
    for a in conn.execute("SELECT surface, kind, speaker, valid_from, valid_until "
                          "FROM aliases WHERE entity_id=? ORDER BY kind", (eid,)):
        window = ""
        if a["valid_from"] or a["valid_until"]:
            window = f"  [{a['valid_from'] or '...'} -> {a['valid_until'] or 'now'}]"
        spk = f"  (by {_name_of(a['speaker'])})" if a["speaker"] else ""
        print(f"    - {a['surface']!r:40} {a['kind']}{spk}{window}")
    n_mentions = conn.execute("SELECT COUNT(*) FROM entity_mentions WHERE entity_id=?",
                              (eid,)).fetchone()[0]
    print(f"  mentions: {n_mentions}")
    print("  claims:")
    for c in conn.execute(
        "SELECT * FROM claims WHERE subject_id=? ORDER BY stated_at", (eid,)):
        label = _claim_label(c, today)
        vu = f" until {c['valid_until'][:10]}" if c["valid_until"] else ""
        conf = f" conf={c['confidence']:.2f}" if c["confidence"] is not None else ""
        drv = " [derived]" if c["derived"] else ""
        print(f"    [{label}] {c['relation']}: {c['object_text']} "
              f"(by {c['asserted_by']}, {c['stated_at']}{vu}){conf}{drv}")
        for ev in json.loads(c["evidence"] or "[]"):
            print(f"        <{ev['row_id']}> {ev['quote']!r}")


def _show_thread(conn: sqlite3.Connection, thread_key: str, today: str) -> None:
    cp = store.get_checkpoint(conn, thread_key)
    if cp:
        print(f"thread {thread_key}: status={cp['status']} blocks_done={cp['blocks_done']} "
              f"watermark={cp['watermark_iso']}")
    claims = conn.execute("SELECT * FROM claims WHERE thread_key=? ORDER BY stated_at",
                          (thread_key,)).fetchall()
    print(f"claims ({len(claims)}):")
    for c in claims:
        label = _claim_label(c, today)
        print(f"  [{label}] {c['subject_text']} — {c['relation']}: {c['object_text']} "
              f"(by {c['asserted_by']}, {c['stated_at']})")
    ctxs = conn.execute("SELECT row_start, row_end, context FROM chunk_contexts "
                        "WHERE thread_key=? ORDER BY row_start LIMIT 30", (thread_key,)).fetchall()
    print(f"contexts ({len(ctxs)} shown, first 30):")
    for c in ctxs:
        print(f"  [{c['row_start']}-{c['row_end']}] {c['context']}")


# --------------------------------------------------------------------------- #
# stats                                                                        #
# --------------------------------------------------------------------------- #

def cmd_stats(args: argparse.Namespace) -> None:
    conn = store.connect(Path(args.memory_db) if args.memory_db else store.MEMORY_DB_PATH)
    try:
        one = lambda q: conn.execute(q).fetchone()[0]
        print("entities:        ", one("SELECT COUNT(*) FROM entities WHERE merged_into IS NULL"))
        print("aliases:         ", one("SELECT COUNT(*) FROM aliases"))
        print("claims (total):  ", one("SELECT COUNT(*) FROM claims"))
        print("claims (valid):  ", one("SELECT COUNT(*) FROM claims WHERE invalidated_at IS NULL"))
        print("claims (derived):", one("SELECT COUNT(*) FROM claims WHERE derived=1"))
        print("mentions:        ", one("SELECT COUNT(*) FROM entity_mentions"))
        print("contexts:        ", one("SELECT COUNT(*) FROM chunk_contexts"))
        print("review queue:    ", one("SELECT COUNT(*) FROM claim_review_queue WHERE resolved=0"))
        blocks = one("SELECT COUNT(*) FROM scan_log")
        if blocks:
            agg = conn.execute(
                "SELECT SUM(n_entities), SUM(n_claims), SUM(evidence_dropped), "
                "SUM(n_claims)+SUM(claims_dropped), AVG(duration_s) FROM scan_log").fetchone()
            new_ent, tot_claims, ev_drop, ev_denom, avg_dur = agg
            print(f"\nblocks scanned:  {blocks}")
            print(f"avg block time:  {avg_dur:.1f}s")
            print(f"new-entity rate: {(new_ent or 0)/blocks*100:.1f} per 100 blocks")
            claims_attempted = one("SELECT SUM(n_claims)+SUM(claims_dropped) FROM scan_log") or 0
            print(f"claims kept:     {tot_claims or 0} (dropped "
                  f"{one('SELECT SUM(claims_dropped) FROM scan_log') or 0})")
            all_ev = conn.execute(
                "SELECT SUM(evidence_dropped) FROM scan_log").fetchone()[0] or 0
            print(f"evidence dropped:{all_ev}")
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# merge (Phase 3 — global entity merge)                                        #
# --------------------------------------------------------------------------- #

def _pair_line(conn: sqlite3.Connection, a: str, b: str) -> str:
    da, db = merge.mini_dossier(conn, a), merge.mini_dossier(conn, b)
    return (f"{da['name']} [{da['kind']}] ({len(da['claims'])} claims)  <->  "
            f"{db['name']} [{db['kind']}] ({len(db['claims'])} claims)")


def cmd_merge(args: argparse.Namespace) -> None:
    _load_openai_env()
    conn = store.connect(Path(args.memory_db) if args.memory_db else store.MEMORY_DB_PATH)
    merge.ensure_schema(conn)
    try:
        if args.unmerge:
            ok = merge.unmerge(conn, args.unmerge)
            print(f"unmerged {args.unmerge}" if ok else
                  f"no reversible merge found for {args.unmerge}")
            return

        embedder = None
        if not args.no_embed:
            from embedder import Embedder
            embedder = Embedder()

        apply = bool(args.apply)  # default is dry-run for safety
        report = merge.run_merge(conn, embedder=embedder, apply=apply,
                                 incremental=args.incremental, threshold=args.threshold)
        cs = report.candidates
        print(f"\ncandidates: {len(cs.auto)} contact_key (auto), "
              f"{len(cs.identity)} ent_me-identity, "
              f"{len(cs.adjudicate)} to adjudicate "
              f"(alias={sum(1 for c in cs.adjudicate if c.source=='alias')}, "
              f"embedding={sum(1 for c in cs.adjudicate if c.source=='embedding')})")

        if cs.auto:
            print("\n[auto-merge — contact_key equality]")
            for c in cs.auto:
                print(f"  {_pair_line(conn, c.a, c.b)}")

        if cs.identity:
            print("\n[ent_me identity absorption — gated on me_identity evidence]")
            for c in cs.identity:
                other = merge._other_side(c)
                name = merge.mini_dossier(conn, other)["name"]
                if c.evidence:
                    print(f"  ABSORB into Me  <-  {name}")
                    for e in c.evidence:
                        print(f"           evidence: {e}")
                else:
                    print(f"  DISTINCT (no user-identity evidence)  Me  <->  {name}")

        print("\n[adjudicated pairs]")
        for v in report.verdicts:
            c = v.candidate
            tag = f"{c.source}" + (f"/{c.score}" if c.source == "embedding" else "")
            print(f"  {v.verdict.upper():<8} ({tag})  {_pair_line(conn, c.a, c.b)}")
            if v.reason:
                print(f"           reason: {v.reason}")

        if apply:
            print(f"\napplied: {len(report.merged)} merge(s)")
            for survivor, loser, reason in report.merged:
                sn = merge.mini_dossier(conn, survivor)["name"]
                print(f"  merged into {sn}: {reason}")
        else:
            print("\n(dry-run — nothing mutated; pass --apply to merge)")
    finally:
        conn.close()


def cmd_merge_review(args: argparse.Namespace) -> None:
    conn = store.connect(Path(args.memory_db) if args.memory_db else store.MEMORY_DB_PATH)
    merge.ensure_schema(conn)
    try:
        rows = conn.execute(
            "SELECT * FROM merge_review WHERE resolved=0 ORDER BY created_at DESC"
        ).fetchall()
        print(f"unsure merge queue: {len(rows)} pair(s)")
        for r in rows:
            print(f"\n  [{r['id'][:8]}] via {r['source']} "
                  f"{('score='+str(r['score'])) if r['source']=='embedding' else ''}")
            print(f"    A: {_dossier_block(conn, r['entity_a'])}")
            print(f"    B: {_dossier_block(conn, r['entity_b'])}")
            if r["reason"]:
                print(f"    adjudicator: {r['reason']}")
    finally:
        conn.close()


def _dossier_block(conn: sqlite3.Connection, entity_id: str) -> str:
    d = merge.mini_dossier(conn, entity_id)
    claims = "; ".join(d["claims"][:3]) if d["claims"] else "no claims"
    return f"{d['name']} [{d['kind']}] — {claims}"


# --------------------------------------------------------------------------- #
# dossiers (Phase 7 — entity summaries from top valid claims, claim-id cached) #
# --------------------------------------------------------------------------- #

DOSSIER_MODEL = "gpt-4o-mini"
DOSSIER_TOPK = 8

DOSSIER_SYSTEM = (
    "You write a one- or two-sentence factual dossier for an entity from someone's "
    "personal message history, given a list of extracted claims. State only what the "
    "claims support: for a person, how they relate to the user and recurring specifics "
    "(role, projects, plans); for a project/event/place/org/thing, what it is and its "
    "current status. Prefer the most recent evidence when claims conflict, and phrase a "
    "state whose end date has passed as a completed transition, never as present-tense "
    "fact. No preamble, no hedging ('it seems'/'appears'), no quotation marks around the "
    "whole answer. Maximum 40 words. If the claims are too thin, output exactly: "
    "'Insufficient claims.'"
)


def _ensure_dossier_schema(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(entities)")}
    if "summary_hash" not in cols:
        conn.execute("ALTER TABLE entities ADD COLUMN summary_hash TEXT")
        conn.commit()


def _dossier_claims(conn: sqlite3.Connection, entity_id: str) -> list[sqlite3.Row]:
    """Top currently-valid claims (most recent first) that feed a dossier."""
    return conn.execute(
        "SELECT * FROM claims WHERE subject_id=? AND invalidated_at IS NULL "
        "ORDER BY stated_at DESC LIMIT ?", (entity_id, DOSSIER_TOPK)).fetchall()


def _dossier_hash(claim_ids: list[str]) -> str:
    payload = "\n".join(sorted(claim_ids))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _dossier_user_prompt(conn: sqlite3.Connection, ent: sqlite3.Row,
                         claims: list[sqlite3.Row], today: str) -> str:
    aliases = [r[0] for r in conn.execute(
        "SELECT DISTINCT surface FROM aliases WHERE entity_id=? AND kind IN "
        "('name','nickname','description') LIMIT 8", (ent["entity_id"],))]
    lines = [f"Entity: {ent['canonical_name']}", f"Kind: {ent['kind']}"]
    if aliases:
        lines.append("Also called: " + ", ".join(aliases))
    lines.append("Claims (newest first):")
    for c in claims:
        label = _claim_label(c, today)
        vu = f" until {c['valid_until'][:10]}" if c["valid_until"] else ""
        lines.append(f"  [{label}] ({c['stated_at'][:10] if c['stated_at'] else '?'}) "
                     f"{c['relation']}: {c['object_text']}{vu}")
    return "\n".join(lines)


def cmd_dossiers(args: argparse.Namespace) -> None:
    _load_openai_env()
    if not os.getenv("OPENAI_API_KEY"):
        print("dossiers: OPENAI_API_KEY not set — cannot summarize", file=sys.stderr)
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    conn = store.connect(Path(args.memory_db) if args.memory_db else store.MEMORY_DB_PATH)
    _ensure_dossier_schema(conn)
    today = date.today().isoformat()
    try:
        if args.entity:
            eid = _resolve_entity_arg(conn, args.entity)
            ents = conn.execute("SELECT * FROM entities WHERE entity_id=?", (eid,)).fetchall() if eid else []
        else:
            # Only entities that actually have valid claims can get a dossier.
            ents = conn.execute(
                "SELECT e.* FROM entities e WHERE e.merged_into IS NULL AND EXISTS "
                "(SELECT 1 FROM claims c WHERE c.subject_id=e.entity_id "
                "AND c.invalidated_at IS NULL) ORDER BY e.first_seen "
                + (f"LIMIT {int(args.limit)}" if args.limit else "")).fetchall()
        written, cached = 0, 0
        for ent in ents:
            claims = _dossier_claims(conn, ent["entity_id"])
            if not claims:
                continue
            h = _dossier_hash([c["claim_id"] for c in claims])
            if ent["summary_hash"] == h and ent["summary"]:
                cached += 1
                continue
            user = _dossier_user_prompt(conn, ent, claims, today)
            try:
                resp = client.chat.completions.create(
                    model=DOSSIER_MODEL, temperature=0.0, max_tokens=90,
                    messages=[{"role": "system", "content": DOSSIER_SYSTEM},
                              {"role": "user", "content": user}])
                summary = (resp.choices[0].message.content or "").strip().strip('"\'')
            except Exception as e:  # noqa: BLE001
                print(f"  {ent['canonical_name']}: dossier failed ({e})", file=sys.stderr)
                continue
            conn.execute(
                "UPDATE entities SET summary=?, summary_hash=?, updated_at=? WHERE entity_id=?",
                (summary, h, datetime.now(timezone.utc).isoformat(), ent["entity_id"]))
            conn.commit()
            written += 1
            print(f"  [{ent['kind']:>8}] {ent['canonical_name']}: {summary}")
        print(f"\ndossiers: wrote {written}, cached {cached}")
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# contextualize (Phase 4 — fold scan contexts into FAISS + FTS)                #
# --------------------------------------------------------------------------- #

def cmd_contextualize(args: argparse.Namespace) -> None:
    memory_db = Path(args.memory_db) if args.memory_db else store.MEMORY_DB_PATH
    metadata_db = Path(args.metadata_db) if args.metadata_db else METADATA_DB_PATH
    report = ctx.run(
        metadata_db=metadata_db, memory_db=memory_db,
        new_only=args.new_only, dry_run=args.dry_run,
    )
    print(f"contextualize ({report['mode']}{', dry-run' if report['dry_run'] else ''})")
    print(f"  chunk_contexts:   {report['contexts_total']} "
          f"(relinked {report['contexts_relinked']}, orphaned {report['contexts_orphaned']}, "
          f"ambiguous ties {report['contexts_ambiguous']})")
    print(f"  chunks w/ context:{report['chunks_with_context']} "
          f"({report['chunks_multi_context']} carry >1 context) of {report['chunks_total']}")
    print(f"  chunks cleared:   {report['chunks_cleared']}")
    print(f"  to re-embed:      {report['chunks_to_reembed']} "
          f"(~{report['est_embed_seconds']}s at assumed MiniLM throughput)")
    if report["dry_run"]:
        print("  (dry-run — no embedding, no FAISS/FTS write)")
        return
    print(f"  chunks_fts rows:  {report['fts_rows']}")
    if "vectors_written" in report:
        print(f"  index rebuilt:    {report['vectors_written']} vectors")
    else:
        print(f"  vectors replaced: {report['vectors_replaced']}")


# --------------------------------------------------------------------------- #
# later-phase stubs                                                            #
# --------------------------------------------------------------------------- #

def _stub(name: str):
    def run(args: argparse.Namespace) -> None:
        print(f"'{name}' is implemented in a later phase.")
    return run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Consolidation scan CLI")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="run/resume the consolidation scan")
    s.add_argument("--thread", help="scan a single thread_key")
    s.add_argument("--fixture", help="path to a fixture file or fixtures/ directory")
    s.add_argument("--max-blocks", type=int, default=None, help="bound the run")
    s.add_argument("--memory-db", help="override memory.db path (throwaway for fixtures)")
    s.add_argument("--include-all-threads", action="store_true",
               help="also scan threads listed in SEMSE_EXCLUDED_THREADS")
    s.set_defaults(func=cmd_scan)

    sh = sub.add_parser("show", help="inspect entities/threads")
    sh.add_argument("--entity", help="entity name or id")
    sh.add_argument("--thread", help="thread_key")
    sh.add_argument("--memory-db")
    sh.set_defaults(func=cmd_show)

    st = sub.add_parser("stats", help="store counts + drift indicators")
    st.add_argument("--memory-db")
    st.set_defaults(func=cmd_stats)

    mg = sub.add_parser("merge", help="global entity merge (dry-run default)")
    mg.add_argument("--apply", action="store_true", help="mutate the store (default: dry-run)")
    mg.add_argument("--incremental", action="store_true",
                    help="only entities created/updated since the last merge run")
    mg.add_argument("--threshold", type=float, default=merge.EMBED_THRESHOLD,
                    help="MiniLM cosine threshold for name neighbors")
    mg.add_argument("--no-embed", action="store_true",
                    help="skip the embedding nominator (alias + contact_key only)")
    mg.add_argument("--unmerge", metavar="ENTITY_ID",
                    help="reverse the most recent merge whose loser was this entity")
    mg.add_argument("--memory-db")
    mg.set_defaults(func=cmd_merge)

    mr = sub.add_parser("merge-review", help="list/inspect the unsure merge queue")
    mr.add_argument("--memory-db")
    mr.set_defaults(func=cmd_merge_review)

    ds = sub.add_parser("dossiers", help="(re)generate entity summaries from valid claims")
    ds.add_argument("--entity", help="only this entity (name or id)")
    ds.add_argument("--limit", type=int, default=None, help="cap number of entities")
    ds.add_argument("--memory-db")
    ds.set_defaults(func=cmd_dossiers)

    cx = sub.add_parser("contextualize",
                        help="fold scan contexts into FAISS + FTS (Phase 4)")
    cx.add_argument("--new-only", action="store_true",
                    help="re-embed only chunks whose context changed (nightly)")
    cx.add_argument("--dry-run", action="store_true",
                    help="report re-link/change counts without embedding or writing")
    cx.add_argument("--memory-db")
    cx.add_argument("--metadata-db")
    cx.set_defaults(func=cmd_contextualize)

    sp = sub.add_parser("nightly", help="nightly (later phase)")
    sp.set_defaults(func=_stub("nightly"))
    return p


def main() -> None:
    _load_openai_env()  # also supplies SEMSE_EXCLUDED_THREADS to every subcommand
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
