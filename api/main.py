"""FastAPI entrypoint."""
from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
load_dotenv()

from .models import SearchRequest, SearchResponse  # noqa: E402
from .search import SearchEngine  # noqa: E402

app = FastAPI(title="Semantic Memory Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # localhost-only API; native client doesn't enforce origin
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

CONTACTS_DIR = Path(__file__).resolve().parent.parent / "indexer" / "data" / "contacts"
SAFE_KEY = re.compile(r"^[a-f0-9]{16}$")

engine: SearchEngine | None = None


@app.on_event("startup")
def _load_engine() -> None:
    global engine
    engine = SearchEngine()


@app.get("/health")
def health() -> dict:
    return {"ok": engine is not None}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    if engine is None:
        raise HTTPException(503, "Search engine not initialized")
    return await engine.search(req)


@app.get("/attachment/{att_id}")
def attachment(att_id: int) -> Response:
    """Serve an iMessage image attachment by its rowid."""
    if engine is None:
        raise HTTPException(503, "engine not ready")
    path = engine.get_image_path(att_id)
    if not path:
        raise HTTPException(404, "not found")
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "file missing")
    data = p.read_bytes()
    media = "application/octet-stream"
    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        media = "image/jpeg"
    elif suffix == ".png":
        media = "image/png"
    elif suffix in (".heic", ".heif"):
        # Many browsers/AppKit can't render HEIC directly; convert to JPEG on the fly.
        try:
            import io
            import pillow_heif
            from PIL import Image
            pillow_heif.register_heif_opener()
            img = Image.open(p).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            media = "image/jpeg"
        except Exception:
            media = "image/heic"
    elif suffix == ".gif":
        media = "image/gif"
    elif suffix == ".tiff":
        media = "image/tiff"
    return Response(content=data, media_type=media,
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/contact-photo/{key}")
def contact_photo(key: str) -> Response:
    """Serve a contact photo blob by its 16-hex key. Falls back to 404 if missing."""
    if not SAFE_KEY.match(key):
        raise HTTPException(400, "invalid key")
    path = CONTACTS_DIR / f"{key}.bin"
    if not path.exists():
        raise HTTPException(404, "not found")
    data = path.read_bytes()
    media = "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        media = "image/png"
    elif data[:4] == b"\xff\xd8\xff\xe0" or data[:4] == b"\xff\xd8\xff\xe1":
        media = "image/jpeg"
    return Response(content=data, media_type=media, headers={"Cache-Control": "public, max-age=86400"})
