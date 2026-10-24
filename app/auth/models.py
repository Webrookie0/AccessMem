"""Domain models for users, roles, and document clearance levels."""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field


class Clearance(IntEnum):
    """Document sensitivity. Higher number = more restricted.

    A user may read any document whose clearance is <= the user's clearance.
    """

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def from_name(cls, name: str) -> Clearance:
        return cls[name.upper()]


# Role -> max clearance the role is allowed to see.
ROLE_CLEARANCE: dict[str, Clearance] = {
    "junior": Clearance.INTERNAL,
    "manager": Clearance.CONFIDENTIAL,
    "csuite": Clearance.RESTRICTED,
    "admin": Clearance.RESTRICTED,
}


class User(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    role: Literal["junior", "manager", "csuite", "admin"]
    tenant_id: str = Field(min_length=1, max_length=128)
    password_hash: str | None = Field(default=None, exclude=True, repr=False)

    @property
    def clearance(self) -> Clearance:
        return ROLE_CLEARANCE[self.role]

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
