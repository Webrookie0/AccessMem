"""RBAC translation layer: turn a User into a vector-store metadata filter.

This is the security-critical piece. NEVER trust user-supplied filters from the
request body — the only allowed filter is the one this module builds from the
authenticated User object.
"""

from .models import Clearance, User


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
    if not isinstance(doc_metadata, dict) or doc_metadata.get("tenant_id") != user.tenant_id:
        return False
    doc_level = doc_metadata.get("clearance_level")
    # Do not coerce strings, floats or booleans into authorization decisions.
    if isinstance(doc_level, bool) or not isinstance(doc_level, int):
        return False
    if doc_level not in {int(level) for level in Clearance}:
        return False
    name = doc_metadata.get("clearance_name")
    if name is not None and name != Clearance(doc_level).name:
        return False
    return doc_level <= int(user.clearance)
