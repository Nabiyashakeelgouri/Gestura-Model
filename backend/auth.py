from datetime import datetime, timedelta, timezone
from typing import Generator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import SessionLocal
from backend.models import User
from backend.schemas import LoginRequest, TokenResponse, UserCreate, UserProfile

router = APIRouter(prefix="/auth", tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def _create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode_access_token(token: str) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        email = payload.get("sub")
        if not email:
            raise credentials_exception
        return email
    except JWTError as exc:
        raise credentials_exception from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication token",
        )

    email = _decode_access_token(credentials.credentials)
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found for token",
        )
    return user


@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(payload: UserCreate, db: Session = Depends(get_db)) -> dict:
    existing_user = (
        db.query(User)
        .filter((User.email == payload.email) | (User.username == payload.username))
        .first()
    )
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email or username already exists",
        )

    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=_hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    return {"message": "User created successfully"}


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account does not exist. Sign up to continue.",
        )

    if not _verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong password or email.",
        )

    token = _create_access_token(user.email)
    return TokenResponse(access_token=token, token_type="bearer")


@router.post("/logout")
def logout() -> dict:
    return {"status": "logged_out", "message": "Logout is handled client-side by discarding the token."}


@router.get("/profile", response_model=UserProfile)
def profile(current_user: User = Depends(get_current_user)) -> UserProfile:
    return UserProfile(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
    )
