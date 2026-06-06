"""In-memory demo user store. Replace with a real database in production."""
from passlib.context import CryptContext

from .models import User

_pwd = CryptContext(schemes=["argon2"], deprecated="auto")

# Pre-computed argon2 hash of "demo" (does not require bcrypt backend initialization)
_DEMO_HASH = "$argon2id$v=19$m=65536,t=3,p=4$sbZ2TskZQyjlHKMUwhiD0A$snno7uZjZ/3skALMrTVVSfNuxttxGJPfwdwA35U2Xhk"


_USERS: dict[str, User] = {
    "alice": User(username="alice", role="junior",  tenant_id="acme",   password_hash=_DEMO_HASH),
    "bob":   User(username="bob",   role="manager", tenant_id="acme",   password_hash=_DEMO_HASH),
    "carol": User(username="carol", role="csuite",  tenant_id="acme",   password_hash=_DEMO_HASH),
    "dave":  User(username="dave",  role="csuite",  tenant_id="globex", password_hash=_DEMO_HASH),
}


def get_user(username: str) -> User | None:
    return _USERS.get(username.lower())


def authenticate(username: str, password: str) -> User | None:
    user = get_user(username)
    if user is None or user.password_hash is None:
        # Run a dummy verify to keep timing constant against username enumeration
        _pwd.verify(password, _DEMO_HASH)
        return None
    if not _pwd.verify(password, user.password_hash):
        return None
    return user


def all_usernames() -> list[str]:
    return sorted(_USERS.keys())
