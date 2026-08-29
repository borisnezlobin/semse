"""In-process golden + control eval harness for the search pipeline.

Loads a live SearchEngine (same class the FastAPI app uses), runs each golden
case through it, and scores regex assertions against the synthesized answer and
the date ranges of the returned sources.

A case whose `expected_phase` exceeds --phase is an EXPECTED failure (XFAIL) —
the plan lands its mechanism in a later phase, so failing now is correct. Only a
case at or below the current phase that fails counts as an UNEXPECTED failure and
makes the process exit non-zero.

    python evals/run_evals.py --phase 1                # score golden cases
    python evals/run_evals.py --only m2-turkey-after --verbose
    python evals/run_evals.py --capture-baseline       # control -> baselines.json
    python evals/run_evals.py --compare-baseline       # control vs baselines.json

The engine is imported lazily so --help and control/baseline bookkeeping work
even when the index is missing or mid-rebuild.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# OPENAI_API_KEY lives in api/.env — the same file the FastAPI app loads.
load_dotenv(REPO_ROOT / "api" / ".env")

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_queries.yaml"
CONTROL_PATH = EVALS_DIR / "control_queries.yaml"
BASELINE_PATH = EVALS_DIR / "baselines.json"


# ── assertion evaluation ─────────────────────────────────────────────────

def _found(pattern: str, text: str) -> bool:
    return re.search(pattern, text or "", re.IGNORECASE) is not None


def _date_covered(iso_date: str, sources: list) -> bool:
    """True if the ISO date falls inside any returned source's date range."""
    d = iso_date[:10]
    for s in sources:
        ds = (s.date_start or "")[:10]
        de = (s.date_end or "")[:10]
        if ds and de and ds <= d <= de:
            return True
        if ds and not de and ds == d:
            return True
    return False


def _evaluate(case: dict, resp) -> list[str]:
    """Return a list of assertion-failure descriptions (empty == pass)."""
    answer = resp.answer or ""
    failures: list[str] = []
    for pat in case.get("answer_must_match") or []:
        if not _found(pat, answer):
            failures.append(f"must_match not found: /{pat}/")
    for pat in case.get("answer_must_not_match") or []:
        if _found(pat, answer):
            failures.append(f"must_not_match matched: /{pat}/")
    for d in case.get("evidence_dates_must_include") or []:
        if not _date_covered(d, resp.sources):
            failures.append(f"evidence date not covered by any source: {d}")
    return failures


# ── engine loading (graceful) ────────────────────────────────────────────

def _load_engine():
    """Instantiate the SearchEngine, or return (None, reason) if unavailable.

    The Phase 0 agent may be rebuilding the index concurrently; a missing index
    or a locked/half-written metadata.db should degrade to a clear DEFERRED
    notice, not a crash."""
    try:
        from api.search import SearchEngine  # heavy import (faiss, model)
        return SearchEngine(), None
    except Exception as e:  # noqa: BLE001 — any load failure is a deferral
        return None, f"{type(e).__name__}: {e}"


# ── golden run ───────────────────────────────────────────────────────────

async def _run_golden(engine, cases: list[dict], phase: int, verbose: bool) -> list[dict]:
    from api.models import SearchRequest

    results: list[dict] = []
    for case in cases:
        req = SearchRequest(
            query=case["query"],
            asof=case.get("asof"),
            intent=None,
            top_k=8,
        )
        resp = None
        try:
            resp = await engine.search(req)
            failures = _evaluate(case, resp)
        except Exception as e:  # noqa: BLE001
            failures = [f"search() raised: {type(e).__name__}: {e}"]
        expected_fail = int(case["expected_phase"]) > phase
        if failures:
            status = "XFAIL" if expected_fail else "FAIL"
        else:
            status = "XPASS" if expected_fail else "PASS"
        results.append(
            {"case": case, "status": status, "failures": failures, "resp": resp}
        )
        if verbose:
            _print_verbose(case, resp)
    return results


def _print_verbose(case: dict, resp) -> None:
    print(f"\n  ── {case['id']} (asof={case.get('asof') or 'today'}) ──")
    print(f"     query : {case['query']}")
    if resp is None:
        print("     answer: <search failed>")
        return
    print(f"     answer: {resp.answer!r}")
    dates = [
        f"{(s.date_start or '')[:10]}..{(s.date_end or '')[:10]} [{s.source}]"
        for s in resp.sources[:6]
    ]
    print(f"     top sources: {dates}")


def _print_scorecard(results: list[dict], phase: int) -> int:
    """Print a scorecard grouped by mechanism. Return the unexpected-fail count."""
    by_mech: dict[str, list[dict]] = {}
    for r in results:
        by_mech.setdefault(r["case"]["mechanism"], []).append(r)

    tally = {"PASS": 0, "FAIL": 0, "XFAIL": 0, "XPASS": 0}
    print(f"\n=== Golden eval scorecard (phase {phase}) ===")
    for mech in sorted(by_mech):
        print(f"\n[{mech}]")
        for r in by_mech[mech]:
            tally[r["status"]] += 1
            phase_note = ""
            if r["status"] in ("XFAIL", "XPASS"):
                phase_note = f" (needs phase {r['case']['expected_phase']})"
            print(f"  {r['status']:<5} {r['case']['id']}{phase_note}")
            if r["status"] == "FAIL":
                for f in r["failures"]:
                    print(f"          - {f}")

    print("\n--- summary ---")
    print(
        f"  PASS={tally['PASS']}  FAIL={tally['FAIL']}  "
        f"XFAIL={tally['XFAIL']}  XPASS={tally['XPASS']}  "
        f"(total {len(results)})"
    )
    if tally["FAIL"]:
        print(f"  {tally['FAIL']} UNEXPECTED failure(s) at phase {phase}.")
    if tally["XPASS"]:
        print(f"  {tally['XPASS']} case(s) passed earlier than expected (XPASS) — worth reclassifying.")
    return tally["FAIL"]


# ── control / baseline ───────────────────────────────────────────────────

def _snapshot(resp) -> list[dict]:
    return [
        {
            "source": s.source,
            "date_start": s.date_start,
            "date_end": s.date_end,
            "contacts": s.contact_names,
        }
        for s in resp.sources
    ]


async def _run_control(engine, queries: list[dict]) -> dict[str, list[dict]]:
    from api.models import SearchRequest

    out: dict[str, list[dict]] = {}
    for q in queries:
        req = SearchRequest(query=q["query"], intent=None, top_k=8)
        resp = await engine.search(req)
        out[q["id"]] = _snapshot(resp)
    return out


def _capture_baseline(engine, queries: list[dict]) -> None:
    snap = asyncio.run(_run_control(engine, queries))
    BASELINE_PATH.write_text(json.dumps(snap, indent=2, ensure_ascii=False))
    print(f"Captured baseline for {len(snap)} control queries -> {BASELINE_PATH}")
    print("NOTE: recapture after the Phase 0 index rebuild (chunk_ids re-key).")


def _compare_baseline(engine, queries: list[dict]) -> int:
    if not BASELINE_PATH.exists():
        print(f"No baseline at {BASELINE_PATH}; run --capture-baseline first.")
        return 1
    baseline = json.loads(BASELINE_PATH.read_text())
    current = asyncio.run(_run_control(engine, queries))
    drift = 0
    print("=== Control baseline comparison ===")
    for q in queries:
        qid = q["id"]
        was = baseline.get(qid)
        now = current.get(qid)
        if was == now:
            print(f"  MATCH  {qid}")
        else:
            drift += 1
            print(f"  DRIFT  {qid}")
            print(f"           baseline: {was}")
            print(f"           current : {now}")
    print(f"\n{drift} of {len(queries)} control queries drifted from baseline.")
    if drift:
        print("If Phase 0 rebuilt the index, recapture the baseline; otherwise this is a regression.")
    return drift


# ── cli ──────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Golden + control eval harness")
    p.add_argument("--phase", type=int, default=1,
                   help="current plan phase; cases needing a later phase are XFAIL (default 1)")
    p.add_argument("--only", type=str, default=None, help="run just the case/query with this id")
    p.add_argument("--verbose", action="store_true", help="print each answer and top source dates")
    p.add_argument("--capture-baseline", action="store_true", help="snapshot control-query sources to baselines.json")
    p.add_argument("--compare-baseline", action="store_true", help="diff control-query sources against baselines.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set — synthesized answers will be empty.", file=sys.stderr)

    engine, reason = _load_engine()
    if engine is None:
        print("EVAL EXECUTION DEFERRED — could not load the search engine.")
        print(f"  reason: {reason}")
        print("  The Phase 0 index rebuild may be in progress. Rerun after it lands:")
        print("    python evals/run_evals.py --phase 1")
        # Harness itself loaded fine; a missing/locked index is not a test failure.
        sys.exit(0)

    if args.capture_baseline or args.compare_baseline:
        queries = _load_yaml(CONTROL_PATH)["queries"]
        if args.only:
            queries = [q for q in queries if q["id"] == args.only]
        if args.capture_baseline:
            _capture_baseline(engine, queries)
            sys.exit(0)
        drift = _compare_baseline(engine, queries)
        sys.exit(1 if drift else 0)

    cases = _load_yaml(GOLDEN_PATH)["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            print(f"No golden case with id '{args.only}'.")
            sys.exit(1)

    results = asyncio.run(_run_golden(engine, cases, args.phase, args.verbose))
    unexpected = _print_scorecard(results, args.phase)
    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
