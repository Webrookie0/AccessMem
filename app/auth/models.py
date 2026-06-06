"""Domain models for users, roles, and document clearance levels."""
from __future__ import annotations

from enum import IntEnum
from pydantic import BaseModel


class Clearance(IntEnum):
    """Document sensitivity. Higher number = more restricted.

    A user may read any document whose clearance is <= the user's clearance.
    """
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def from_name(cls, name: str) -> "Clearance":
        return cls[name.upper()]


# Role -> max clearance the role is allowed to see.
ROLE_CLEARANCE: dict[str, Clearance] = {
    "junior": Clearance.INTERNAL,
    "manager": Clearance.CONFIDENTIAL,
    "csuite": Clearance.RESTRICTED,
    "admin": Clearance.RESTRICTED,
}


class User(BaseModel):
    username: str
    role: str
    tenant_id: str
    # password_hash is NOT exposed via this model when serialised to clients
    password_hash: str | None = None

    @property
    def clearance(self) -> Clearance:
        return ROLE_CLEARANCE.get(self.role, Clearance.PUBLIC)

    def public_dict(self) -> dict:
        return {
            "username": self.username,
            "role": self.role,
            "tenant_id": self.tenant_id,
            "clearance": self.clearance.name,
        }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict
