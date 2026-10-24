"""JWT authentication backed by the current server-side user record."""

from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from ..config import get_settings
from .models import User
from .users import get_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(user: User) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": user.username,
            "role": user.role,
            "tenant_id": user.tenant_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    settings = get_settings()
    exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={
                "require_exp": True,
                "require_iat": True,
                "require_sub": True,
                "require_iss": True,
                "require_aud": True,
            },
        )
    except JWTError as exc:
        raise exception from exc
    username = payload.get("sub")
    if not isinstance(username, str) or not username:
        raise exception
    user = get_user(username)
    if (
        user is None
        or payload.get("tenant_id") != user.tenant_id
        or payload.get("role") != user.role
    ):
        raise exception
    return user
