from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models import User

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.delete("/user/delete")
def delete_user(username: str, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.username == username).first()

    if not user:
        return {"message": "User not found"}

    db.delete(user)
    db.commit()

    return {"message": "User deleted successfully"}