"""Pydantic models for the search API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceTag = Literal["imessage", "mail", "hyperspell", "image"]


class QueryIntent(BaseModel):
    """Structured intent parsed from the raw query by the on-device router.

    Sent from the Swift client. The server uses these as hard retrieval filters
    and replaces the embedded text with `topic` so filler words ("picture of",
    "email about") don't dilute the semantic vector.
    """
    topic: str = ""
    sources: list[str] = []
    contacts: list[str] = []
    must_have_attachment: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 8
    intent: QueryIntent | None = None


class ChunkMessage(BaseModel):
    sender: str
    is_from_me: bool
    text: str
    date_iso: str
    contact_key: str | None = None
    known: bool = True
    is_best_match: bool = False  # Per-query, set in API layer for UI anchoring


class SourceResult(BaseModel):
    source: SourceTag | str
    contact_names: list[str]
    date_start: str
    date_end: str
    score: float
    messages: list[ChunkMessage] = []
    subject: str | None = None
    chat_title: str | None = None
    snippet: str = ""
    # Image-source fields (populated when source == "image")
    image_url: str | None = None     # /attachment/{id} on this server
    image_caption: str | None = None # original message text from the image's parent message
    attachment_id: int | None = None


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    query_ms: int
