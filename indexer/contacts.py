"""Resolve phone numbers and email addresses to real Contacts via macOS AddressBook.

Reads every AddressBook-v22.abcddb under ~/Library/Application Support/AddressBook/
(per-source: iCloud, On My Mac, Exchange, etc.) and builds an in-memory index of
key → (display_name, photo_jpeg_bytes_or_None).

Phone numbers are normalized to digits-only with an optional leading '+', so that
'(408) 315-8094', '+1 408 315 8094', and '4083158094' all collide on the same key.
"""
from __future__ import annotations

import glob
import hashlib
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ADDRESSBOOK_GLOB = str(Path.home() / "Library" / "Application Support" / "AddressBook" / "**" / "AddressBook-v22.abcddb")

CONTACTS_DIR = Path(__file__).parent / "data" / "contacts"

UNKNOWN_LABEL = "Unknown"


@dataclass(frozen=True)
class Contact:
    name: str
    phones: tuple[str, ...]
    emails: tuple[str, ...]
    photo_bytes: bytes | None


_PHONE_DIGITS = re.compile(r"\D+")


def normalize_phone(raw: str | None) -> str | None:
    """Reduce a phone number to digits (with optional leading +). Returns None if empty."""
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    has_plus = cleaned.lstrip().startswith("+")
    digits = _PHONE_DIGITS.sub("", cleaned)
    if not digits:
        return None
    # If it's a US 10-digit number, normalize with leading 1.
    if len(digits) == 10 and not has_plus:
        digits = "1" + digits
    return f"+{digits}"


def normalize_email(raw: str | None) -> str | None:
    if not raw:
        return None
    e = raw.strip().lower()
    return e or None


def normalize_handle(handle: str | None) -> str | None:
    """iMessage handles can be either a phone or an email; route accordingly."""
    if not handle:
        return None
    if "@" in handle:
        return normalize_email(handle)
    return normalize_phone(handle)


class ContactResolver:
    """Loads all AddressBook DBs once, then answers handle → (name, photo) lookups."""

    def __init__(self) -> None:
        self._by_phone: dict[str, Contact] = {}
        self._by_email: dict[str, Contact] = {}
        self._photo_cache_keys: dict[str, str] = {}  # name → file key
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for db_path in glob.glob(ADDRESSBOOK_GLOB, recursive=True):
            try:
                self._ingest_db(Path(db_path))
            except Exception as e:  # one corrupt source shouldn't kill everything
                print(f"  contacts: skipping {db_path}: {e}")
        self._loaded = True

    def _ingest_db(self, path: Path) -> None:
        # Copy to temp to avoid touching the live SQLite (and to handle WAL files cleanly).
        tmp = Path(tempfile.mkdtemp(prefix="semse_ab_"))
        dest = tmp / "ab.db"
        shutil.copy2(path, dest)
        for sidecar in (f"{path.name}-wal", f"{path.name}-shm"):
            sc = path.parent / sidecar
            if sc.exists():
                shutil.copy2(sc, tmp / sidecar.replace(path.name, "ab.db"))
        conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        try:
            self._read_records(conn)
        finally:
            conn.close()

    def _read_records(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        # Map record PK → display name + photo. Photos live on ZABCDRECORD itself,
        # in ZTHUMBNAILIMAGEDATA (small) or ZIMAGEDATA (full) BLOB columns.
        records: dict[int, tuple[str, bytes | None]] = {}
        for pk, first, last, nickname, organization, thumb, full in cur.execute(
            """
            SELECT Z_PK, ZFIRSTNAME, ZLASTNAME, ZNICKNAME, ZORGANIZATION,
                   ZTHUMBNAILIMAGEDATA, ZIMAGEDATA
            FROM ZABCDRECORD
            """
        ):
            name = _build_display_name(first, last, nickname, organization)
            if not name:
                continue
            photo = bytes(thumb) if thumb else (bytes(full) if full else None)
            records[pk] = (name, photo)

        # Phones
        for owner, full_number in cur.execute(
            "SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER WHERE ZFULLNUMBER IS NOT NULL"
        ):
            rec = records.get(owner)
            if not rec:
                continue
            key = normalize_phone(full_number)
            if not key:
                continue
            self._by_phone.setdefault(
                key,
                Contact(name=rec[0], phones=(key,), emails=(), photo_bytes=rec[1]),
            )

        # Emails
        for owner, address in cur.execute(
            "SELECT ZOWNER, ZADDRESS FROM ZABCDEMAILADDRESS WHERE ZADDRESS IS NOT NULL"
        ):
            rec = records.get(owner)
            if not rec:
                continue
            key = normalize_email(address)
            if not key:
                continue
            self._by_email.setdefault(
                key,
                Contact(name=rec[0], phones=(), emails=(key,), photo_bytes=rec[1]),
            )

    # -- Public lookup API --

    def resolve(self, handle: str | None) -> Contact | None:
        if not handle:
            return None
        if "@" in handle:
            return self._by_email.get(normalize_email(handle) or "")
        return self._by_phone.get(normalize_phone(handle) or "")

    def display_for(self, handle: str | None, fallback: str = UNKNOWN_LABEL) -> str:
        c = self.resolve(handle)
        if c:
            return c.name
        if not handle:
            return fallback
        # Last-ditch: for unresolved phones, show last 4 digits, never the raw number.
        if "@" not in handle:
            digits = _PHONE_DIGITS.sub("", handle)
            if len(digits) >= 4:
                return f"{fallback} (•{digits[-4:]})"
        return fallback

    def photo_key(self, handle: str | None) -> str | None:
        """Stable filename-safe key for a contact's photo, suitable for /contact-photo/{key}."""
        c = self.resolve(handle)
        if not c or not c.photo_bytes:
            return None
        return _name_to_key(c.name)

    def export_photos(self, out_dir: Path = CONTACTS_DIR) -> int:
        """Write photo blobs to disk so the API can serve them. Returns count written.

        Apple stores blobs with a 1-byte prefix:
          0x01 = inline JPEG follows (strip the byte and write as .jpg)
          0x02 = UUID reference to external file (skipped for now)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, bytes] = {}
        for c in {*self._by_phone.values(), *self._by_email.values()}:
            if not c.photo_bytes:
                continue
            key = _name_to_key(c.name)
            seen.setdefault(key, c.photo_bytes)
        written = 0
        for key, data in seen.items():
            if not data:
                continue
            prefix = data[0:1]
            if prefix == b"\x01" and len(data) > 1:
                (out_dir / f"{key}.bin").write_bytes(data[1:])
                written += 1
            elif data[:3] == b"\xff\xd8\xff":
                (out_dir / f"{key}.bin").write_bytes(data)
                written += 1
            else:
                # External-reference (0x02) or unknown format — skip silently.
                continue
        return written


def _build_display_name(first, last, nickname, organization) -> str | None:
    if nickname:
        return str(nickname).strip()
    parts = [p for p in (first, last) if p]
    if parts:
        return " ".join(str(p).strip() for p in parts)
    if organization:
        return str(organization).strip()
    return None


def _name_to_key(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    r = ContactResolver()
    r.load()
    print(f"phones: {len(r._by_phone)}  emails: {len(r._by_email)}")
    sample = list(r._by_phone.items())[:5]
    for k, v in sample:
        photo = "yes" if v.photo_bytes else "no"
        print(f"  {k:18s} → {v.name}  (photo: {photo})")
    n = r.export_photos()
    print(f"exported {n} photos to {CONTACTS_DIR}")
