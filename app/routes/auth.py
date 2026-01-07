from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.models import SignupRequest
from app.database import users_collection
from app.utils.system_specs import get_system_specs

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup")
def signup(data: SignupRequest):
    if users_collection.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="User already exists")

    specs = get_system_specs()

    user = {
        "username": data.username,
        "email": data.email,
        "password": data.password,
        "specs": specs,
        "status": "offline",
        "created_at": datetime.utcnow()
    }

    users_collection.insert_one(user)

    return {
        "message": "Signup successful",
        "node": {
            "username": data.username,
            "specs": specs
        }
    }
