from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class WordInput(BaseModel):
    word: str


class TextInput(BaseModel):
    text: str