from pydantic import BaseModel, EmailStr
from typing import Dict, Any

class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    specs: Dict[str, Any] | None = None

class SigninRequest(BaseModel):
    email: EmailStr
    password: str