from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ..auth.jwt import create_access_token, get_current_user
from ..auth.models import TokenResponse, User
from ..auth.users import authenticate
from ..services.audit import log_event

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    user = authenticate(form.username, form.password)
    if user is None:
        log_event("login_failed", username=form.username)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    log_event("login_ok", username=user.username, role=user.role, tenant=user.tenant_id)
    token = create_access_token(user)
    return TokenResponse(access_token=token, user=user.public_dict())


@router.get("/me")
def me(user: User = Depends(get_current_user)) -> dict:
    return user.public_dict()
