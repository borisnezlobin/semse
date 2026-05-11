# Semantic Memory Search — CLAUDE.md

## Vision

A Spotlight/Raycast-style search interface over your entire personal communication history.
One search bar. Natural language queries. Synthesized answer + cited source cards in ~2 seconds.
No chat interface. No onboarding flow. Just search.

---

## Architecture

Two decoupled phases:

```
PHASE 1 — Indexing (run once, ~20–30 min)
  parse_sources.py → chunk → embed (local, sentence-transformers) → FAISS index + metadata.db

PHASE 2 — Runtime
  FastAPI server (localhost:8000) — embeds query, searches FAISS, fetches context, calls OpenAI
  Next.js app  (localhost:3000) — the Spotlight UI, calls the FastAPI server
  Hyperspell API — queried at search time (not pre-indexed), results merged server-side
```

---

## Tech Stack (do not substitute anything)

**Indexing / backend:**
- Python 3.11+
- `sentence-transformers` — model: `all-MiniLM-L6-v2` (384-dim, fast, local)
- `faiss-cpu` — IVFFlat index with nlist=256 for 1M+ vectors; FlatIP for <100k
- `sqlite3` (stdlib) — for reading iMessage and Apple Mail; also for metadata store
- `beautifulsoup4` + `lxml` — HTML stripping from emails
- `python-fastapi` + `uvicorn` — the search API
- `openai` Python SDK — gpt-4o-mini for synthesis (fast, cheap)
- `tqdm` — progress bars during indexing
- `numpy`

**Frontend:**
- Next.js 14 (App Router), TypeScript strict mode
- Tailwind CSS (no component library)
- Framer Motion — search result animations only
- `@phosphor-icons/react` — all icons
- `openai` npm package if needed for any client-side calls (unlikely)

---

## File Structure

```
/
├── CLAUDE.md
├── indexer/
│   ├── parse_imessage.py       # reads ~/Library/Messages/chat.db
│   ├── parse_mail.py           # reads Apple Mail Envelope Index + .emlx files
│   ├── chunker.py              # sliding window chunker, shared logic
│   ├── embedder.py             # sentence-transformers wrapper, batch embed
│   ├── build_index.py          # orchestrates all sources → FAISS index + metadata.db
│   └── requirements.txt
├── api/
│   ├── main.py                 # FastAPI app, /search endpoint
│   ├── search.py               # embed query → FAISS → context expansion → Hyperspell → OpenAI
│   ├── hyperspell.py           # Hyperspell query-time integration
│   └── models.py               # Pydantic models
├── app/                        # Next.js App Router
│   ├── page.tsx                # renders <SearchPage />
│   ├── layout.tsx
│   └── globals.css
├── components/
│   ├── SearchPage.tsx          # full-screen layout with centered modal
│   ├── SearchModal.tsx         # the spotlight panel — input + results
│   ├── SynthesisCard.tsx       # AI-generated answer at top of results
│   ├── SourceCard.tsx          # individual cited source
│   └── PlatformIcon.tsx        # routes to correct Phosphor icon per source
├── lib/
│   └── api.ts                  # typed fetch wrapper → FastAPI
├── .env.local
└── package.json
```

---

## Phase 1 — Indexing

### Chunking Strategy

All sources use the same chunk shape:
```python
@dataclass
class Chunk:
    chunk_id: str          # uuid
    source: str            # "imessage" | "mail" | "hyperspell_gmail" etc.
    contact_names: list[str]
    date_start: str        # ISO 8601
    date_end: str
    text: str              # the text to embed — concatenated messages
    row_ids: list[int]     # original DB row IDs for context expansion at query time
```

**iMessage chunks:** sliding window of 20 messages, stride 10, grouped by `chat_id`. Sort by date ASC within each chat before chunking. Include `is_from_me` labels so the LLM knows who said what ("Me: ...\nSarah: ...").

**Apple Mail chunks:** one chunk per thread (group by thread ID). If a single email body exceeds 800 tokens, split it. Strip HTML before chunking.

### iMessage Parser (`indexer/parse_imessage.py`)

Source DB: `~/Library/Messages/chat.db` — copy to a temp path first, do NOT write to the original.

Key tables and the exact query to use:
```sql
SELECT
  m.rowid,
  m.text,
  m.date,
  m.is_from_me,
  h.id         AS contact_id,
  c.rowid      AS chat_id,
  c.display_name,
  c.chat_identifier
FROM message m
LEFT JOIN handle h         ON m.handle_id       = h.rowid
LEFT JOIN chat_message_join cmj ON m.rowid      = cmj.message_id
LEFT JOIN chat c           ON cmj.chat_id        = c.rowid
WHERE m.text IS NOT NULL
  AND length(trim(m.text)) > 0
ORDER BY chat_id, m.date ASC
```

Date conversion (iMessage dates are nanoseconds since 2001-01-01 00:00:00 UTC):
```python
import datetime
MAC_EPOCH_OFFSET = 978307200  # seconds between Unix epoch and Mac epoch
def imessage_date_to_iso(ns: int) -> str:
    unix_ts = ns / 1_000_000_000 + MAC_EPOCH_OFFSET
    return datetime.datetime.utcfromtimestamp(unix_ts).isoformat()
```

Filter out messages where `text` starts with `\ufffd` (reaction messages) or is only a single emoji.

### Apple Mail Parser (`indexer/parse_mail.py`)

**Step 1 — Envelope Index (metadata):**
Path: `~/Library/Mail/V*/MailData/Envelope Index` (glob for the versioned folder, take the most recent).
This is a SQLite database. First run `PRAGMA table_info(messages)` and `.tables` to inspect — the schema varies across macOS versions. The key columns you need: subject, sender address, date received (Unix timestamp or similar), thread ID or In-Reply-To, and a path/reference to the `.emlx` file.

**Step 2 — .emlx files:**
Located at `~/Library/Mail/V*/[account]/[mailbox]/Messages/*.emlx`.
Format: first line is an integer (byte count of RFC 2822 content), followed by the raw email, followed by Apple plist XML (ignore the plist).

Parse with:
```python
import email
from email import policy as email_policy

def parse_emlx(path: str) -> email.message.Message:
    with open(path, "rb") as f:
        f.readline()  # skip byte count
        raw = f.read()
    # strip trailing plist — find "<?xml" and truncate
    xml_start = raw.find(b"<?xml")
    if xml_start != -1:
        raw = raw[:xml_start]
    return email.message_from_bytes(raw, policy=email_policy.default)
```

Extract body: prefer `text/plain` part. If only `text/html`, use BeautifulSoup to strip tags. Decode with the charset from the Content-Type header.

Filter: skip any email where the sender domain is in a blocklist of known newsletter/marketing domains (unsubscribe.com patterns, noreply@, no-reply@, notifications@, etc.) OR where the email has never been replied to AND is older than a threshold.

### Embedder (`indexer/embedder.py`)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,  # for cosine similarity via inner product
            convert_to_numpy=True,
        )
```

### Index Builder (`indexer/build_index.py`)

1. Parse all sources → list of `Chunk`
2. Embed all chunk `.text` fields in one batched pass
3. Build FAISS index:
   - If n_chunks < 100_000: use `faiss.IndexFlatIP` (exact, normalized vectors → cosine similarity)
   - If n_chunks >= 100_000: use `faiss.IndexIVFFlat(quantizer, 384, 256, faiss.METRIC_INNER_PRODUCT)`, train on the vectors, then add
4. Save index to `indexer/data/index.faiss`
5. Save chunk metadata (everything except the embedding) to `indexer/data/metadata.db` (SQLite), table `chunks`:
   ```sql
   CREATE TABLE chunks (
     chunk_id TEXT PRIMARY KEY,
     source TEXT,
     contact_names TEXT,  -- JSON array
     date_start TEXT,
     date_end TEXT,
     text TEXT,
     row_ids TEXT         -- JSON array of ints
   );
   ```
6. Print final stats: n_chunks per source, total, index size.

---

## Phase 2 — Search API

### FastAPI App (`api/main.py`)

Single endpoint. Load FAISS index and metadata DB at startup (global singletons).

```python
@app.post("/search")
async def search(req: SearchRequest) -> SearchResponse:
    ...
```

```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = 8

class SourceResult(BaseModel):
    source: str          # "imessage" | "mail" | "hyperspell"
    contact_names: list[str]
    date_start: str
    date_end: str
    snippet: str         # first 280 chars of chunk text
    score: float

class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    query_ms: int
```

### Search Logic (`api/search.py`)

```
1. Embed query with same model (load once at startup)
2. Search local FAISS → top_k results → fetch chunk metadata from metadata.db
3. Call hyperspell.search(query) → get Hyperspell results (see below)
4. Merge: deduplicate by content similarity, take top 6 total (prefer diversity of sources)
5. Build context string:
   For each result:
     [SOURCE: iMessage | CONTACT: Sarah | DATE: Jan 14 2025]
     <chunk text>
     ---
6. Call OpenAI:
   system: "You are a personal memory assistant. Answer the user's question using ONLY
            the provided conversation excerpts. Be specific and concise. If the answer
            spans multiple sources, synthesize them. Cite sources inline as [iMessage·Sarah]
            or [Gmail·Alex]. If you cannot find the answer, say so directly."
   user: f"Question: {query}\n\nExcerpts:\n{context}"
   model: gpt-4o-mini
   max_tokens: 400
7. Return answer + sources list
```

### Hyperspell Integration (`api/hyperspell.py`)

Query Hyperspell at search time — do NOT pre-index. Check `https://docs.hyperspell.com` for the exact query endpoint and authentication. The result should be a list of text excerpts with source metadata (platform, sender, date). Normalize these into the same `SourceResult` shape. If the Hyperspell API call fails or times out (>3s), continue with local results only — never block the response.

Use `HYPERSPELL_API_KEY` from environment.

---

## UI Design Specification

### Concept

macOS Spotlight — but warmer, rounder, more personal. The kind of tool that feels like it's been on your Mac for years. Not a web app that happens to look like a Mac app. Dark theme only.

### Layout

Full viewport. Dark backdrop (`rgba(0,0,0,0.55)`) with `backdrop-filter: blur(8px)` over a subtle noise texture background. The modal panel floats centered — both horizontally and vertically, shifted 15% upward from center (same as Spotlight).

Panel dimensions: `680px` wide, `min-height: 64px`, expands downward as results arrive. `border-radius: 22px`. Background: `#1c1c1e` (not pure black). Subtle `box-shadow: 0 32px 80px rgba(0,0,0,0.7), 0 0 0 0.5px rgba(255,255,255,0.08)`.

### Search Input

Full-width inside panel, `padding: 18px 20px 18px 52px`. Font: `SF Pro Text` with `-apple-system` fallback, `text-size: 18px`, `font-weight: 400`. No border, no background (transparent). Caret color: `#0A84FF` (macOS blue). Placeholder: `"Search your conversations…"` in `rgba(255,255,255,0.3)`.

Phosphor `MagnifyingGlass` icon (weight: `light`, size: 20) at left, `rgba(255,255,255,0.4)`.

Thin `1px` divider below the input (`rgba(255,255,255,0.06)`) — only visible when results are present.

### Synthesis Card (top result)

Appears first, before source cards. Background: `rgba(255,255,255,0.04)`. Border: `0.5px solid rgba(255,255,255,0.08)`. `border-radius: 14px`. `padding: 14px 16px`. 

Top row: Phosphor `Sparkle` icon (size 13, color `#0A84FF`) + label `"Answer"` in `10px uppercase tracking-wider rgba(255,255,255,0.4)`.
Body: the synthesized answer in `14px`, `line-height: 1.6`, `rgba(255,255,255,0.85)`. Inline citations styled as `[iMessage · Sarah]` in `rgba(255,255,255,0.4) text-xs`.

### Source Cards

Stacked below synthesis card with `8px` gap. Each card: `padding: 10px 14px`, `border-radius: 12px`, `background: transparent`, hover state: `rgba(255,255,255,0.05)`. 

Layout:
```
[PlatformIcon 20px]  [Contact Name bold 13px]             [date muted 11px]
                     [snippet text 12px muted 2 lines max]
```

Platform icons (use Phosphor, weight `fill`):
- iMessage → `ChatCircle` tinted `#30d158` (green)
- Apple Mail → `EnvelopeSimple` tinted `#0A84FF` (blue)
- Slack → `Hash` tinted `#E01E5A`
- Gmail → `EnvelopeSimple` tinted `#EA4335`
- Notion → `FileText` tinted `rgba(255,255,255,0.6)`

Contact name: `rgba(255,255,255,0.9)`, `font-weight: 500`.
Date + platform label: `rgba(255,255,255,0.35)`, `11px`. Format: `"iMessage · Jan 14, 2025"`.
Snippet: `rgba(255,255,255,0.5)`, `12.5px`, max 2 lines, `overflow: hidden`.

### Animations (Framer Motion)

- Panel entrance: `initial={{ opacity: 0, scale: 0.97, y: -8 }}` → `animate={{ opacity: 1, scale: 1, y: 0 }}`, `duration: 0.15`, `ease: [0.32, 0.72, 0, 1]`
- Results container: `AnimatePresence`
- Each source card: staggered `delay: index * 0.04`, `initial={{ opacity: 0, y: 6 }}` → `animate={{ opacity: 1, y: 0 }}`
- Synthesis card: `delay: 0`, same translate animation

### Loading State

While waiting for the API response: show 3 placeholder cards with a shimmer animation (CSS `@keyframes` scanning highlight, not a spinner). The shimmer should be subtle — `rgba(255,255,255,0.04)` to `rgba(255,255,255,0.08)`.

### Keyboard Behavior

- `ArrowDown` / `ArrowUp`: navigate source cards (highlight with `rgba(255,255,255,0.06)` background)
- `Escape`: clear input and dismiss results
- `Enter` on a highlighted card: expand the full chunk text inline (accordion, `AnimatePresence` height animation)
- Do not implement anything else

---

## API Client (`lib/api.ts`)

```typescript
export interface SourceResult {
  source: "imessage" | "mail" | "hyperspell";
  contact_names: string[];
  date_start: string;
  date_end: string;
  snippet: string;
  score: number;
}

export interface SearchResponse {
  answer: string;
  sources: SourceResult[];
  query_ms: number;
}

export async function search(query: string): Promise<SearchResponse> {
  const res = await fetch("http://localhost:8000/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 8 }),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}
```

Debounce search calls by 300ms in the component. Trigger search when query length >= 4 characters.

---

## Environment Variables

`.env.local` (Next.js):
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

`indexer/.env` + `api/.env`:
```
OPENAI_API_KEY=...
HYPERSPELL_API_KEY=...
```

---

## Implementation Order

Build and verify in this exact order. Do not proceed to the next step until the current one works end-to-end.

1. `indexer/requirements.txt` + `pip install`
2. `indexer/parse_imessage.py` — test by printing 10 sample chunks to stdout
3. `indexer/chunker.py` — unit test the sliding window logic
4. `indexer/embedder.py` — test by embedding 5 strings and checking shape
5. `indexer/build_index.py` — run on iMessage only first, verify index loads and returns results
6. `api/main.py` + `api/search.py` — wire up FAISS search, test with curl
7. OpenAI synthesis in `api/search.py` — verify the answer is coherent
8. Next.js scaffold — `app/page.tsx`, `components/SearchPage.tsx`, `components/SearchModal.tsx` with static UI (no API calls yet)
9. Wire `lib/api.ts` → `SearchModal.tsx` — live search working
10. `components/SynthesisCard.tsx` + `components/SourceCard.tsx` — full result display
11. Animations (Framer Motion)
12. `indexer/parse_mail.py` — add Apple Mail to the index
13. `api/hyperspell.py` — add Hyperspell at query time

---

## Strict DO NOT Rules

- Do not add authentication of any kind
- Do not add a database ORM or Prisma
- Do not use any CSS framework other than Tailwind
- Do not use any component library (shadcn, MUI, Radix, etc.)
- Do not add a chat interface — this is a search tool, not a chatbot
- Do not write to `~/Library/Messages/chat.db` — copy it first, read only
- Do not use any font other than the system font stack (`-apple-system, BlinkMacSystemFont, "SF Pro Text"`)
- Do not add routing — this is a single page
- Do not use React Query, SWR, or any data fetching library — use plain `fetch` with `useEffect`
- Do not round border-radius below `12px` on any card or below `22px` on the main panel
- Do not use purple, blue gradients, or any color associated with generic AI products — the accent color is `#0A84FF` (macOS blue) only

---

## Demo Queries to Verify

Once fully built, these queries should return meaningful results (assuming real data is indexed):

- `"what events has [name] mentioned wanting to go to recently"`
- `"summarize that conversation I had about [topic] — I forget who with"`
- `"has anyone recommended a [thing] to me"`
- `"what did [person] say about [project]"`
