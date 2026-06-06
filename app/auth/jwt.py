"""JWT issue + verify, plus the FastAPI dependency that hydrates the User."""
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from ..config import get_settings
from .models import User
from .users import get_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(user: User) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.username,
        "role": user.role,
        "tenant_id": user.tenant_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    settings = get_settings()
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        if not username:
            raise creds_exc
    except JWTError:
        raise creds_exc

    user = get_user(username)
    if user is None:
        raise creds_exc
    # Cross-check tenant claim against current store (defends against stale tokens after a tenant move)
    if payload.get("tenant_id") != user.tenant_id:
        raise creds_exc
    return user
