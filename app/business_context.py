"""
Business context manager — stores and retrieves user-submitted business
context suggestions and admin-approved context.

Storage: a single JSON file at ``<project_root>/data/business_context.json``.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_CONTEXT_FILE = _DATA_DIR / "business_context.json"


def _ensure_file() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _CONTEXT_FILE.exists():
        _CONTEXT_FILE.write_text(
            json.dumps({"suggestions": [], "approved": []}, indent=2),
            encoding="utf-8",
        )
    return _CONTEXT_FILE


def _load() -> dict[str, Any]:
    path = _ensure_file()
    return json.loads(path.read_text(encoding="utf-8"))


def _save(data: dict[str, Any]) -> None:
    path = _ensure_file()
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── Public API ────────────────────────────────────────────────────────────────

def add_suggestion(text: str, submitted_by: str = "user") -> dict:
    """Add a new pending business-context suggestion from a user."""
    data = _load()
    entry = {
        "id": str(uuid.uuid4())[:8],
        "text": text.strip(),
        "submitted_by": submitted_by,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    data["suggestions"].append(entry)
    _save(data)
    return entry


def get_suggestions(status: str | None = None) -> list[dict]:
    """Return suggestions, optionally filtered by status."""
    data = _load()
    if status is None:
        return data["suggestions"]
    return [s for s in data["suggestions"] if s.get("status") == status]


def approve_suggestion(suggestion_id: str, admin_text: str | None = None) -> bool:
    """
    Approve a suggestion: move it to the approved list.
    *admin_text* lets the admin edit the wording before approving.
    """
    data = _load()
    for s in data["suggestions"]:
        if s["id"] == suggestion_id and s["status"] == "pending":
            s["status"] = "approved"
            s["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            approved_entry = {
                "id": s["id"],
                "text": (admin_text or s["text"]).strip(),
                "approved_at": s["reviewed_at"],
                "original_text": s["text"],
                "submitted_by": s.get("submitted_by", "user"),
            }
            data["approved"].append(approved_entry)
            _save(data)
            return True
    return False


def reject_suggestion(suggestion_id: str) -> bool:
    """Reject a pending suggestion."""
    data = _load()
    for s in data["suggestions"]:
        if s["id"] == suggestion_id and s["status"] == "pending":
            s["status"] = "rejected"
            s["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            _save(data)
            return True
    return False


def get_approved_context() -> list[dict]:
    """Return all approved business-context entries."""
    return _load().get("approved", [])


def remove_approved(entry_id: str) -> bool:
    """Remove an entry from the approved context list."""
    data = _load()
    before = len(data["approved"])
    data["approved"] = [a for a in data["approved"] if a["id"] != entry_id]
    if len(data["approved"]) < before:
        _save(data)
        return True
    return False


def update_approved_text(entry_id: str, new_text: str) -> bool:
    """Update the text of an approved context entry."""
    data = _load()
    for a in data["approved"]:
        if a["id"] == entry_id:
            a["text"] = new_text.strip()
            _save(data)
            return True
    return False


def format_approved_for_prompt() -> str:
    """
    Build a prompt-ready string from all approved business context.
    Returns empty string when there is no approved context.
    """
    approved = get_approved_context()
    if not approved:
        return ""
    lines = ["## Business Context (domain knowledge from stakeholders)"]
    for i, entry in enumerate(approved, 1):
        lines.append(f"{i}. {entry['text']}")
    return "\n".join(lines)
