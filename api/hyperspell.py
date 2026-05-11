"""Hyperspell query-time integration. Always non-blocking; failures are silently swallowed."""
from __future__ import annotations

import asyncio
import os

import httpx

from .models import SourceResult

HYPERSPELL_URL = "https://api.hyperspell.com/memories/query"
TIMEOUT_SECONDS = 3.0


async def search(query: str, top_k: int = 8) -> list[SourceResult]:
    api_key = os.getenv("HYPERSPELL_API_KEY")
    if not api_key:
        return []
    payload = {
        "query": query,
        "answer": False,
        "options": {"max_results": top_k},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(HYPERSPELL_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, asyncio.TimeoutError):
        return []
    except Exception:
        return []

    results: list[SourceResult] = []
    for doc in data.get("documents") or []:
        meta = doc.get("metadata") or {}
        title = doc.get("title") or "(untitled)"
        source = doc.get("source") or "hyperspell"
        date = meta.get("created_at") or meta.get("last_modified") or meta.get("indexed_at") or ""
        snippet = title
        if meta.get("url"):
            snippet = f"{title} — {meta['url']}"
        results.append(
            SourceResult(
                source="hyperspell",
                contact_names=[_pretty_source(source)],
                date_start=date or "",
                date_end=date or "",
                snippet=snippet[:280],
                score=float(doc.get("score") or 0.0),
            )
        )
    return results


def _pretty_source(source: str) -> str:
    return {
        "google_mail": "Gmail",
        "gmail_actions": "Gmail",
        "google_calendar": "Google Calendar",
        "google_drive": "Google Drive",
        "slack": "Slack",
        "notion": "Notion",
        "github": "GitHub",
        "reddit": "Reddit",
        "dropbox": "Dropbox",
        "box": "Box",
        "microsoft_teams": "Microsoft Teams",
        "web_crawler": "Web",
        "vault": "Vault",
        "trace": "Trace",
    }.get(source, source.title())
