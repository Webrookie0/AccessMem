"""RBAC translation layer: turn a User into a vector-store metadata filter.

This is the security-critical piece. NEVER trust user-supplied filters from the
request body — the only allowed filter is the one this module builds from the
authenticated User object.
"""
from .models import User, Clearance


def build_access_filter(user: User) -> dict:
    """Return a Chroma `where` clause restricting the search to:
      * the user's own tenant, AND
      * documents whose clearance level is <= the user's own clearance.
    """
    return {
        "$and": [
            {"tenant_id": {"$eq": user.tenant_id}},
            {"clearance_level": {"$lte": int(user.clearance)}},
        ]
    }


def can_user_read(user: User, doc_metadata: dict) -> bool:
    """Defence-in-depth: re-check after retrieval that every returned doc
    is actually allowed for this user. Any mismatch means a bug in the
    filter and we drop the doc."""
    if doc_metadata.get("tenant_id") != user.tenant_id:
        return False
    doc_level = int(doc_metadata.get("clearance_level", Clearance.RESTRICTED))
    return doc_level <= int(user.clearance)
