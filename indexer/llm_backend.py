"""LLM backend adapter for the consolidation scan.

The scan needs strict-JSON extraction from a local model at $0. Ollama is the default
(qwen3.5-class, 4-bit) driven through /api/chat with `format` set to a JSON schema so the
server grammar-constrains the output. An OpenAI-shaped stub implements the same interface
so the backend becomes a config swap once measured quality justifies it.

The one non-obvious knob: qwen3.5 is a *thinking* model. Left in thinking mode it emits a
long unconstrained chain-of-thought before the JSON, which is both slow (90s+ for a trivial
call) and wasteful. We pass `think: False` to suppress it — measured warm calls then land
in the low single-digit seconds for small outputs.

Backend selection: env SEMSE_SCAN_BACKEND (ollama | openai), default ollama.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Completion:
    """A single structured completion plus the telemetry the scan_log wants."""
    data: dict
    tokens_in: int
    tokens_out: int
    duration_s: float
    model: str
    repaired: bool = False


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Strict-JSON schema for OpenAI Structured Outputs                             #
# --------------------------------------------------------------------------- #

def to_strict_schema(schema: dict) -> dict:
    """Rewrite a permissive JSON schema into the closed form OpenAI `strict` mode requires.

    Strict mode demands `additionalProperties:false` on every object and EVERY property
    listed in `required`; optionality is expressed only by allowing null in a field's type.
    The scan's OUTPUT_SCHEMA already makes the genuinely-optional fields nullable, so closing
    the objects here is a faithful, lossless transform — the model still may emit empty arrays
    or nulls, it just may not omit keys entirely.
    """
    if not isinstance(schema, dict):
        return schema
    node = dict(schema)
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        node["properties"] = {k: to_strict_schema(v) for k, v in node["properties"].items()}
        node["required"] = list(node["properties"].keys())
        node["additionalProperties"] = False
    if "items" in node:
        node["items"] = to_strict_schema(node["items"])
    return node


def _strip_code_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences despite a schema — peel them."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> str:
    """Best-effort recovery of the first balanced {...} object from noisy output."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text[start:]


def _try_parse(content: str) -> dict | None:
    for candidate in (content, _strip_code_fences(content)):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(_extract_json_object(_strip_code_fences(content)))
    except Exception:
        return None


class OllamaBackend:
    def __init__(self) -> None:
        self.host = os.getenv("SEMSE_SCAN_HOST", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("SEMSE_SCAN_MODEL", "qwen3.5:latest")
        self.keep_alive = os.getenv("SEMSE_SCAN_KEEPALIVE", "30m")
        self.timeout = float(os.getenv("SEMSE_SCAN_TIMEOUT", "600"))
        self.max_retries = int(os.getenv("SEMSE_SCAN_RETRIES", "2"))
        # Must fit the full prompt AND the generated output; if prompt+output overflows the
        # window, Ollama shifts context mid-generation and evicts the system prompt, which
        # silently empties later sections or sends generation into a degenerate loop.
        # 100-msg blocks prompt at ~7-9.5k tokens and emit ~2.5-3.5k, so 16384 leaves margin.
        self.num_ctx = int(os.getenv("SEMSE_SCAN_NUM_CTX", "16384"))
        # Hard output cap: a runaway generation gets truncated (then JSON-repaired or
        # retried) instead of grinding until the HTTP timeout on every retry.
        self.num_predict = int(os.getenv("SEMSE_SCAN_NUM_PREDICT", "4096"))

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def complete_verbose(self, system: str, user: str, json_schema: dict) -> Completion:
        base_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": json_schema,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict},
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t = time.time()
            try:
                resp = self._post(base_payload)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue
            dt = time.time() - t
            content = resp.get("message", {}).get("content", "")
            tokens_in = int(resp.get("prompt_eval_count") or 0)
            tokens_out = int(resp.get("eval_count") or 0)
            data = _try_parse(content)
            if data is not None:
                return Completion(data, tokens_in, tokens_out, dt, self.model)
            # Repair round-trip: hand the malformed text back and ask for valid JSON only.
            repaired = self._repair(system, content, json_schema)
            if repaired is not None:
                return Completion(repaired.data, tokens_in + repaired.tokens_in,
                                  tokens_out + repaired.tokens_out, dt + repaired.duration_s,
                                  self.model, repaired=True)
            last_err = LLMError(f"unparseable output: {content[:200]!r}")
        raise LLMError(f"Ollama completion failed after retries: {last_err}")

    def _repair(self, system: str, bad_content: str, json_schema: dict) -> Completion | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content":
                    "You output only valid JSON matching the schema. No prose, no fences."},
                {"role": "user", "content":
                    "The following output was not valid JSON. Re-emit it as a single valid "
                    "JSON object matching the schema, fixing only the syntax:\n\n" + bad_content},
            ],
            "stream": False,
            "format": json_schema,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict},
        }
        t = time.time()
        try:
            resp = self._post(payload)
        except Exception:
            return None
        content = resp.get("message", {}).get("content", "")
        data = _try_parse(content)
        if data is None:
            return None
        return Completion(data, int(resp.get("prompt_eval_count") or 0),
                          int(resp.get("eval_count") or 0), time.time() - t, self.model)

    def complete(self, system: str, user: str, json_schema: dict) -> dict:
        return self.complete_verbose(system, user, json_schema).data


CAPTION_SYSTEM = (
    "You caption one image or video frame shared in a personal text message. Reply with ONE "
    "short factual sentence describing only what is visibly present — the main subjects, "
    "objects, setting, and any clearly legible on-screen text. If it is plainly a screenshot "
    "(app, chat, terminal, document), say so and name what is on screen. No preamble, no "
    "'this image shows', no speculation about intent."
)

_VIDEO_FRAME_MAX_SIDE = 1024


def _video_frame_to_jpeg(path: str) -> bytes | None:
    """Grab one representative frame from a video as JPEG bytes via ffmpeg's thumbnail filter."""
    if not shutil.which("ffmpeg"):
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        out = tf.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-vf", f"thumbnail,scale='min({_VIDEO_FRAME_MAX_SIDE},iw)':-1", "-frames:v", "1", out],
            check=True, timeout=60,
        )
        data = Path(out).read_bytes()
        return data or None
    except Exception:
        return None
    finally:
        Path(out).unlink(missing_ok=True)


def _image_to_jpeg(path: str, max_side: int = 1024) -> bytes | None:
    """Load any still image (incl. HEIC) and re-encode as a downscaled JPEG for the vision API."""
    try:
        from PIL import Image
        try:
            import pillow_heif  # HEIC/HEIF support — most iMessage photos are HEIC
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception:
        return None


def attachment_to_jpeg(path: str, mime: str | None) -> bytes | None:
    """Normalize an image or video attachment to JPEG bytes the vision model can read."""
    p = str(path)
    is_video = (mime or "").startswith("video/") or p.lower().endswith(
        (".mov", ".mp4", ".m4v", ".avi", ".3gp", ".mkv", ".webm"))
    if is_video:
        return _video_frame_to_jpeg(p)
    return _image_to_jpeg(p)


class OpenAIBackend:
    """OpenAI structured-output extraction + gpt-4o-mini vision captioning.

    Kept import-light so the module loads without the openai package installed; the client
    is only constructed on first use. Caption usage is accumulated on the instance so the
    experiment can price the vision pass separately from the extraction pass.
    """

    def __init__(self) -> None:
        self.model = os.getenv("SEMSE_SCAN_OPENAI_MODEL", "gpt-4o-mini")
        self.vision_model = os.getenv("SEMSE_CAPTION_MODEL", "gpt-4o-mini")
        self.vision_detail = os.getenv("SEMSE_CAPTION_DETAIL", "low")
        self._client = None
        self._strict = os.getenv("SEMSE_OPENAI_STRICT", "1") != "0"
        # Running caption-cost telemetry.
        self.caption_calls = 0
        self.caption_tokens_in = 0
        self.caption_tokens_out = 0

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # deferred — optional dependency
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _response_format(self, json_schema: dict) -> dict:
        schema = to_strict_schema(json_schema) if self._strict else json_schema
        return {"type": "json_schema",
                "json_schema": {"name": "scan_block", "schema": schema, "strict": self._strict}}

    def complete_verbose(self, system: str, user: str, json_schema: dict) -> Completion:
        client = self._ensure_client()
        t = time.time()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format=self._response_format(json_schema),
                temperature=0,
            )
        except Exception as e:  # noqa: BLE001
            # A closed schema OpenAI won't accept (rare) shouldn't kill the run — fall back
            # to non-strict json_schema for this and all later calls, then retry once.
            if self._strict and "schema" in str(e).lower():
                self._strict = False
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format=self._response_format(json_schema),
                    temperature=0,
                )
            else:
                raise
        content = resp.choices[0].message.content or "{}"
        data = _try_parse(content) or {}
        usage = resp.usage
        return Completion(data, getattr(usage, "prompt_tokens", 0),
                          getattr(usage, "completion_tokens", 0), time.time() - t, self.model)

    def complete(self, system: str, user: str, json_schema: dict) -> dict:
        return self.complete_verbose(system, user, json_schema).data

    def caption(self, path: str, mime: str | None = None) -> str | None:
        """A short factual caption for an image/video attachment, or None if unreadable."""
        jpeg = attachment_to_jpeg(path, mime)
        if not jpeg:
            return None
        b64 = base64.b64encode(jpeg).decode()
        is_video = (mime or "").startswith("video/") or str(path).lower().endswith(
            (".mov", ".mp4", ".m4v", ".avi", ".3gp"))
        kind_hint = "a frame from a video" if is_video else "an image"
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.vision_model, temperature=0, max_tokens=70,
            messages=[
                {"role": "system", "content": CAPTION_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Caption this ({kind_hint} shared in a text)."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}", "detail": self.vision_detail}},
                ]},
            ],
        )
        usage = resp.usage
        self.caption_calls += 1
        self.caption_tokens_in += getattr(usage, "prompt_tokens", 0)
        self.caption_tokens_out += getattr(usage, "completion_tokens", 0)
        text = (resp.choices[0].message.content or "").strip().strip('"')
        return text or None


def get_backend():
    name = os.getenv("SEMSE_SCAN_BACKEND", "ollama").lower()
    if name == "openai":
        return OpenAIBackend()
    return OllamaBackend()


def get_captioner():
    """A vision captioner for attachment enrichment.

    Captions describe fixed pixels, so they are model-independent and cached forever — the
    local (Ollama) extraction backend reuses OpenAI-authored captions for free. Selectable
    via SEMSE_CAPTION_BACKEND (default openai); returns None when no captioner is available
    so callers degrade to text-only cleanly.
    """
    name = os.getenv("SEMSE_CAPTION_BACKEND", "openai").lower()
    if name in ("", "none", "off"):
        return None
    if name == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return OpenAIBackend()
    return None


if __name__ == "__main__":
    backend = get_backend()
    schema = {
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
    }
    c = backend.complete_verbose(
        "Respond as JSON.", "Say hello to the consolidation scan.", schema
    )
    print(f"model={c.model} in={c.tokens_in} out={c.tokens_out} {c.duration_s:.1f}s")
    print(c.data)
