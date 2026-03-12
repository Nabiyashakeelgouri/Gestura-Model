from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.auth import get_current_user, get_db
from backend.models import AccountDeletionToken, User
from backend.schemas import DeleteRequestResponse

router = APIRouter(prefix="/user", tags=["user"])


def _delete_user_now(user: User, db: Session) -> DeleteRequestResponse:
    db.query(AccountDeletionToken).filter(
        AccountDeletionToken.user_id == user.id
    ).delete(synchronize_session=False)
    db.delete(user)
    db.commit()

    return DeleteRequestResponse(
        status="deleted",
        message="Your account has been deleted successfully.",
    )


@router.post("/delete", response_model=DeleteRequestResponse)
def delete_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeleteRequestResponse:
    return _delete_user_now(current_user, db)


# Backward-compatible alias for older frontend calls.
@router.post("/delete/request", response_model=DeleteRequestResponse)
def request_account_deletion(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeleteRequestResponse:
    return _delete_user_now(current_user, db)
