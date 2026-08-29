"""Prompts and I/O shape for the consolidation scan.

Design note on the schema: the local Ollama build (0.23.x) treats `format` as a
valid-JSON guarantee only — it does NOT grammar-constrain property names (verified: the
model happily renames keys). So the authoritative contract is the worked example baked
into the system prompt; `OUTPUT_SCHEMA` is passed to `format` as a light guard and stays
correct in case a future backend enforces it. Post-processing tolerates missing/extra keys.

TODAY_IN_SCAN is the block's last message date. The model resolves every relative date
against the per-message timestamps in the transcript and never reasons past TODAY_IN_SCAN,
which prevents anachronistic claims when scanning old history.
"""
from __future__ import annotations

import json
import re

OBJ_REPLACEMENT = "￼"  # U+FFFC — iMessage's inline attachment placeholder (￼)

# A genuine schema mirroring the target shape. additionalProperties stays permissive so
# the current loose backend still parses; a stricter backend would enforce these names.
OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string"},
                    "kind": {"type": "string"},
                    "aliases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "surface": {"type": "string"},
                                "kind": {"type": "string"},
                                "speaker": {"type": ["string", "null"]},
                                "valid_from": {"type": ["string", "null"]},
                                "valid_until": {"type": ["string", "null"]},
                            },
                            "required": ["surface", "kind"],
                        },
                    },
                },
                "required": ["canonical_name", "kind"],
            },
        },
        "contexts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_start": {"type": "integer"},
                    "row_end": {"type": "integer"},
                    "context": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["row_start", "row_end", "context"],
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object_text": {"type": "string"},
                    "object_ref": {"type": ["string", "null"]},
                    "asserted_by": {"type": "string"},
                    "stated_at": {"type": "string"},
                    "valid_from": {"type": ["string", "null"]},
                    "valid_until": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "derived": {"type": "boolean"},
                    "supersedes_prior": {"type": "boolean"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_id": {"type": "integer"},
                                "quote": {"type": "string"},
                            },
                            "required": ["row_id", "quote"],
                        },
                    },
                },
                "required": ["subject", "relation", "object_text", "asserted_by",
                             "stated_at", "evidence"],
            },
        },
    },
    "required": ["entities", "contexts", "claims"],
}


SYSTEM_PROMPT_TEMPLATE = """You are a consolidation engine reading one person's chat history in \
chronological order, block by block, like a human catching up on a conversation. You are \
given a running memory (the WORKING STATE) of ongoing situations and, for each block, emit a \
strict-JSON update capturing the entities, situations, and claims in it.

You are reading as of TODAY_IN_SCAN (the date of the last message in this block). Never \
assert anything about events after that date; resolve every relative date against the \
message that contains it.

OUTPUT: a single JSON object with EXACTLY these keys and no others:
{
  "entities": [
    {
      "canonical_name": "the clearest name for this thing",
      "kind": "person | project | event | activity | place | org | thing | pet | media",
      "aliases": [
        {"surface": "exact phrase used", "kind": "name|nickname|description|handle",
         "speaker": "who used it, or null", "valid_from": "YYYY-MM-DD or null",
         "valid_until": "YYYY-MM-DD or null"}
      ]
    }
  ],
  "contexts": [
    {"row_start": <int>, "row_end": <int>,
     "context": "1-2 sentence situation summary naming the entities involved",
     "entities": ["entity name", ...]}
  ],
  "claims": [
    {
      "subject": "entity name or short phrase the claim is about",
      "relation": "snake_case predicate (e.g. located_in, scheduled_at, renamed_to, owes)",
      "object_text": "the value / complement, resolved to plain English",
      "object_ref": "another entity name if the object IS an entity, else null",
      "asserted_by": "the speaker who stated it (a name, or Me)",
      "stated_at": "YYYY-MM-DD the claim was made",
      "valid_from": "YYYY-MM-DD or null",
      "valid_until": "YYYY-MM-DD or null — set when the text bounds the state",
      "confidence": 0.0-1.0,
      "derived": true|false,
      "supersedes_prior": true|false,
      "evidence": [{"row_id": <int>, "quote": "verbatim substring of that message"}]
    }
  ]
}

RULES

Entities (recognition gate):
- Only create an entity for something genuinely referred to and worth remembering: a \
person, a project, a named or datable event, a recurring activity, a place, a pet, a piece \
of media. Do NOT mint entities for one-off small talk, function words, or vague nouns.
- BIND, DON'T DUPLICATE. If a new surface form refers to something already in the WORKING \
STATE or KNOWN ENTITIES, reuse that entity and add the new form as an alias — do not create \
a second entity. This includes: abbreviations ("mvc" IS "multivariable calc"); a handle or \
contact label and a real name being the same person ("alarm" IS "Alara" -> one entity, with \
"alarm" as a kind:handle alias); descriptive references before a name is coined ("the email \
digest thing" then "Quillstream" -> one entity).
- Alias kinds: "name" (proper name), "nickname", "description" (a role/phrase like "my \
english teacher" — set speaker to whoever's viewpoint it is; "my teacher" from Alara != from \
Me), "handle" (a sender label / contact string / unknown-number label).
- Alias validity windows: when a name changes over time, set valid_until on the old surface \
and valid_from on the new one so either era's vocabulary resolves to the one entity.

Claims (assertion, not truth):
%%CLAIM_BUDGET_RULE%%
- Recognition gate: a claim is worth emitting only if it states a fact a reader would want \
recalled later — a plan, a state, a preference, a commitment, an event, a relationship, a \
piece of shared content. Skip pure phatic talk ("almost there", "lol", "same"), and skip \
anything you could not quote support for. Ask of every claim: "would the speaker stand \
behind this as a fact?" — if not, either mark it low-confidence or drop it. Precision beats \
recall: one wrong or noisy claim poisons more than one missed claim saves.
- A claim records what was ASSERTED and by whom. Set asserted_by to the speaker.
- Speaker integrity. Each transcript line is tagged [row | time | speaker]; you always know \
who said what. Attribute a claim to the person whose OWN words carry it, and cite evidence \
rows spoken BY that person. In a question-and-answer exchange the answer belongs to the \
answerer, not the asker: if A asks "do you want X?" and B replies "yes", the wanting is B's \
— never credit A with B's answer, and never credit B with A's question. You may add the \
other speaker's turn as EXTRA evidence only when it is genuinely theirs and it does not flip \
who the subject is. When unsure who owns a claim, prefer the row the subject spoke themselves.
- Every claim needs >=1 evidence item whose quote is copied VERBATIM (character for \
character) from the cited row's message text. If you cannot quote it exactly, drop the claim.
- Resolve relative dates to absolute ones against the message timestamp: "back Tuesday", \
"tomorrow", "next week", "the 13th", "weekend after AP exams" -> a concrete YYYY-MM-DD, \
chaining through dates already known. Put the resolved date in object_text/valid_until.
- valid_until: when the text bounds a state ("staying till Jan 9", "flight home Friday the \
13th"), set valid_until to that resolved date. Put the bound ON the state claim itself: if \
someone is away and a return date is known, the located_in / staying_at claim carries \
valid_until = the return date — never leave the state unbounded with the date parked only in \
a separate returns_on claim. When a stay/trip/presence has BOTH a vague duration ("the whole \
week") AND an explicit return/end date ("flight home Friday the 13th"), bound the state with \
the explicit dated event — more reliable than arithmetic on a vague duration. Bind each \
bound to the correct subject: the same date can bound two people's states in opposite \
directions ("she's back the 15th" vs "I fly out the 15th" — her trip ENDS then, mine STARTS \
then). A move/reschedule is a NEW claim that supersedes the old one.
- Location discipline. A located_in / staying_at / lives_in claim means a SETTLED place held \
for a non-trivial stretch — someone's home, a city they are visiting, the venue of an event. \
MOTION and TRANSIT are not location claims: "going down Kingsley from Bryant", "driving to \
Mountain View", "on my way", "heading home", "walking from A to B", "just left", "5 minutes \
away" all describe travel, not residence — a street or road named only as the PATH between \
two points is never a located_in. Skip transit outright unless the DESTINATION is itself \
durable and worth recording; if so, record the arrival as relation heading_to with \
valid_until set to that same day, never as a standing located_in. Never infer a standing \
location from an in-transit message.
- supersedes_prior: set true when this claim replaces an earlier state of the SAME subject \
and relation (a reschedule, a rename, a status change, a discharged commitment). It will \
invalidate the prior claim, never delete it.
- Commitments, ratified plans, and questions-awaiting-reply ARE claims (relation like \
plans_to, committed_to, asked, owes). Their fulfillment/cancellation/answer is a later claim \
that supersedes them.
- Literal vs banter: playful, figurative, or hyperbolic claims get confidence <=0.3. \
Triggers: mock accusations or insults between friends ("BK GROOMS CHILDREN BTW", "X is a \
bum school"), trash-talk comparisons and joke superlatives ("black rock got nothing on \
bear rock"), all-caps exclamations played for laughs, drive-by "btw" jabs, figurative \
debts ("owe you 9 bobas", "two mistletoe kisses"), sarcastic rhetorical questions, \
meme-speak. Ask: would the speaker stand behind this as a fact? If the statement would be \
absurd or defamatory taken literally from a friend, it is banter — cap confidence at 0.3, \
or skip it entirely if it is not worth remembering. Reserve confidence >=0.7 for \
statements plainly meant literally.
- The evidence quote must by itself SUPPORT the claim: a reader seeing only the quote \
should reach the same subject-relation-object conclusion. A quote that merely mentions \
the subject does not establish the relation (a comment about what a place is like is not \
evidence that anyone is going there). If no quote in the transcript establishes the \
relation, skip the claim or cap confidence at 0.4.
- Prefer quotes that CARRY the claim's content. A bare acknowledgment ("yeah", "ok", "fr") \
proves something was agreed, not what. When the content sits in a question or an earlier \
message and the confirmation in a reply ("so you're back the 15th?" / "yeah"), cite BOTH \
rows as evidence — the content row and the confirmation row.
- Depth over vagueness. When a message conveys real content, capture the CONTENT, not a \
generic stand-in. Do not collapse a rich event into a bland relation like has_contact_with \
or mentioned when something specific actually happened — record what was shared, planned, \
decided, or reported. Reserve bare contact/mention relations for when there is genuinely \
nothing more specific to say.
- Image and video captions. A line reading "[image: ...]" or "[video: ...]" is a \
machine-generated description of a picture or clip attached to that message — treat it as \
reported visual content, on equal footing with text. When the message says who sent or \
shared it (e.g. the words "Grandma sent" beside "[video: a hedgehog in the grass]"), emit a \
content claim about WHAT was shared (subject = the sender, relation like shared or sent, \
object_text = the described content), not merely that a message occurred. Quote the message's \
own WORDS as evidence (the caption describes pixels, not words, but it is a valid substring to \
quote when nothing else carries the content). Here the unresolved-deixis rule relaxes: an \
"it"/"this" whose referent is the captioned image IS now resolvable — resolve it to the \
caption's content.
- Aggregate/statistical assertions are OPINIONS about data, not independent ground truth — \
set derived:true on them. Triggers: any claim quantifying frequency, counts, proportions, or \
universals — "every single day", "always", "never", "most days", "more than 20% of days", \
"we've done X N times", "usually", "all the time". If the claim summarizes HOW OFTEN or HOW \
MANY rather than a single event, it is derived:true.
- Unresolved deixis: only bind referents you can resolve from the transcript or WORKING \
STATE. If "it"/"this" points at an UNCAPTIONED attachment or an unknown antecedent, do NOT \
guess — omit the claim. (When a "[image: ...]"/"[video: ...]" caption is present, the \
referent IS resolved — see the caption rule above.)
- Quoted/borrowed material — song lyrics, memes, movie lines, copypasta (recognizable by a \
sudden shift in register or by your own knowledge) — asserts NOTHING. Emit no claim from it, \
UNLESS the surrounding conversation takes it up literally and confirms a real situation \
(then the claim comes from the uptake message, not the quote).
- "Me" lines are valid evidence for the user's own facts — do not discard them.

Language:
- Write claims, relations, object_text, and contexts in ENGLISH even when the messages are \
in another language. But keep every evidence quote VERBATIM in its original script.

Contexts (REQUIRED — always emit at least one; aim for 2 to 6 per block):
- Segment the block into its handful of distinct situations and summarize each so anaphoric \
messages ("it's at 7") become findable by their referent. 1-2 sentences each, naming the \
entities, with row_start/row_end covering that situation's message rows. Group adjacent tiny \
messages that belong to one situation under a SINGLE context spanning their row range — do \
not write one context per message. An empty contexts array is never acceptable.

Working memory:
- The WORKING STATE you are given is your memory of ongoing situations from earlier blocks. \
Use it to resolve references ("the extension", "she", "the launch") to the right entity. You \
do not emit it — it is maintained for you from the store.
"""


def _claim_budget_rule(claim_cap: int | None) -> str:
    """The one claim-quantity rule, tuned to the block's cap.

    The cap is a guardrail, not a target: with a capable model we lean on the recognition
    gate + confidence rather than a blunt ceiling, so a NO-cap run should not degrade into
    padding. A blunt low ceiling (the old value of 10) truncated real facts on dense blocks.
    """
    if claim_cap is None:
        return (
            "- There is no fixed limit on how many claims a block may hold — capture every "
            "load-bearing fact in it. This is NOT license to pad: a claim earns its place "
            "only by passing the recognition gate below. Emitting ten solid facts is better "
            "than twenty padded with trivia; emitting thirty is right when the block truly "
            "holds thirty facts worth recalling."
        )
    return (
        f"- Emit up to {claim_cap} claims for this block — a generous ceiling, not a target. "
        f"Do not pad toward it: every claim must pass the recognition gate below. If a block "
        f"genuinely holds more than {claim_cap} facts worth recalling, keep the {claim_cap} "
        f"most load-bearing and drop the rest."
    )


def build_system_prompt(claim_cap: int | None = None) -> str:
    """The scan system prompt with the claim-quantity rule bound to the block's cap.

    A plain replace (not str.format) because the prompt embeds a literal JSON example full
    of braces that format() would misread as fields.
    """
    return SYSTEM_PROMPT_TEMPLATE.replace("%%CLAIM_BUDGET_RULE%%", _claim_budget_rule(claim_cap))


# Default prompt (no numeric cap) for callers/tests that don't thread a cap through.
SYSTEM_PROMPT = build_system_prompt(None)


def estimate_tokens(text: str) -> int:
    """Cheap char/4 heuristic — good enough for the 250-msg / 6k-token block budget."""
    return len(text) // 4


def apply_caption(text: str, caption_tags: str) -> str:
    """Fold an attachment caption into a message: drop the ￼ placeholders and lead with the
    caption tag(s) so the model reads the picture/clip as content rather than a blank glyph."""
    body = re.sub(r"\s+", " ", (text or "").replace(OBJ_REPLACEMENT, " ")).strip()
    return f"{caption_tags} {body}".strip() if body else caption_tags


def format_transcript(
    messages: list[dict], row_ids: list[int], captions: dict[int, str] | None = None
) -> str:
    """One line per message: [row_id | timestamp | speaker] text.

    When `captions` maps a row_id to a "[image: ...]"/"[video: ...]" tag, that message's ￼
    attachment placeholder is replaced by the caption so the model sees the visual content.
    """
    lines = []
    for msg, rid in zip(messages, row_ids):
        ts = (msg.get("date_iso") or "")[:19]
        speaker = "Me" if msg.get("is_from_me") else (msg.get("sender") or "Unknown")
        text = (msg.get("text") or "").replace("\n", " ").strip()
        if captions and rid in captions:
            text = apply_caption(text, captions[rid])
        lines.append(f"[{rid} | {ts} | {speaker}] {text}")
    return "\n".join(lines)


def build_user_prompt(
    *,
    today_in_scan: str,
    participants: list[str],
    working_state: list[dict],
    paged_entities: list[dict],
    transcript: str,
) -> str:
    state_str = json.dumps(working_state, ensure_ascii=False, indent=1) if working_state else "[]"
    known = [
        {"canonical_name": e["canonical_name"], "kind": e.get("kind"),
         "aliases": [a["surface"] for a in e.get("aliases", [])]}
        for e in paged_entities
    ]
    known_str = json.dumps(known, ensure_ascii=False, indent=1) if known else "[]"
    return (
        f"TODAY_IN_SCAN: {today_in_scan}\n"
        f"THREAD PARTICIPANTS: {', '.join(participants)}\n\n"
        f"KNOWN ENTITIES (already in the store — bind to these, do not duplicate):\n{known_str}\n\n"
        f"WORKING STATE (your memory carried from earlier blocks):\n{state_str}\n\n"
        f"TRANSCRIPT (this block, chronological):\n{transcript}\n\n"
        f"Emit the JSON update for this block now."
    )
