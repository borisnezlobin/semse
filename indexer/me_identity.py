"""Discover the user's own email addresses and phone numbers.

Sources, in order of trust:
  1. AddressBook record flagged as "me" (ZME=1 / Z22_ME=1)
  2. macOS Accounts4.sqlite — every account's username (most are emails)
  3. The MEMORY_USER_EMAILS env variable (comma-separated override)

We need this so that mail messages the user sent get marked `is_from_me=True`
(otherwise the user shows up as a "person they met"), and so the LLM can
distinguish the user from contacts in chunks where both appear.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

from contacts import ADDRESSBOOK_GLOB, normalize_email, normalize_phone
import glob


def _from_addressbook() -> tuple[set[str], set[str]]:
    """Return (emails, phones) flagged as the user's own AddressBook record."""
    emails: set[str] = set()
    phones: set[str] = set()
    for db_path in glob.glob(ADDRESSBOOK_GLOB, recursive=True):
        try:
            tmp = Path(tempfile.mkdtemp(prefix="semse_me_"))
            dest = tmp / "ab.db"
            shutil.copy2(db_path, dest)
            conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
            try:
                me_pks = [
                    r[0]
                    for r in conn.execute(
                        "SELECT Z_PK FROM ZABCDRECORD WHERE ZME=1 OR Z22_ME=1"
                    )
                ]
                if not me_pks:
                    continue
                placeholders = ",".join("?" * len(me_pks))
                for (addr,) in conn.execute(
                    f"SELECT ZADDRESS FROM ZABCDEMAILADDRESS WHERE ZOWNER IN ({placeholders})",
                    me_pks,
                ):
                    e = normalize_email(addr)
                    if e:
                        emails.add(e)
                for (num,) in conn.execute(
                    f"SELECT ZFULLNUMBER FROM ZABCDPHONENUMBER WHERE ZOWNER IN ({placeholders})",
                    me_pks,
                ):
                    p = normalize_phone(num)
                    if p:
                        phones.add(p)
            finally:
                conn.close()
        except Exception:
            continue
    return emails, phones


def _from_accounts_db() -> set[str]:
    """Pull every email-shaped username from macOS Accounts4.sqlite."""
    emails: set[str] = set()
    db_path = Path.home() / "Library" / "Accounts" / "Accounts4.sqlite"
    if not db_path.exists():
        return emails
    try:
        tmp = Path(tempfile.mkdtemp(prefix="semse_acc_"))
        dest = tmp / "a.db"
        shutil.copy2(db_path, dest)
        conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        try:
            for (username,) in conn.execute(
                "SELECT ZUSERNAME FROM ZACCOUNT WHERE ZUSERNAME LIKE '%@%'"
            ):
                e = normalize_email(username)
                if e:
                    emails.add(e)
        finally:
            conn.close()
    except Exception:
        pass
    return emails


_BLOCKLIST_DOMAINS = {"local", "localhost", "icloud.com", "me.com"}


def discover() -> tuple[set[str], set[str]]:
    """Return (emails, phones) belonging to the user. Idempotent and cheap."""
    emails, phones = _from_addressbook()
    emails |= _from_accounts_db()
    override = os.getenv("MEMORY_USER_EMAILS", "").strip()
    if override:
        for e in override.split(","):
            n = normalize_email(e)
            if n:
                emails.add(n)
    override_phones = os.getenv("MEMORY_USER_PHONES", "").strip()
    if override_phones:
        for p in override_phones.split(","):
            n = normalize_phone(p)
            if n:
                phones.add(n)
    return emails, phones


def is_user_handle(handle: str | None, user_emails: set[str], user_phones: set[str]) -> bool:
    if not handle:
        return False
    if "@" in handle:
        return (normalize_email(handle) or "") in user_emails
    return (normalize_phone(handle) or "") in user_phones


if __name__ == "__main__":
    emails, phones = discover()
    print(f"emails: {sorted(emails)}")
    print(f"phones: {sorted(phones)}")
