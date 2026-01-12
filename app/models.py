from pydantic import BaseModel, EmailStr, Field
from typing import Dict, Any, Optional, List


class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    specs: Optional[Dict[str, Any]] = None


class SigninRequest(BaseModel):
    email: EmailStr
    password: str


class CreateSessionRequest(BaseModel):
    # number of peers (excluding the creator)
    num_peers: int = Field(..., ge=1, le=100)
    # optional session name / metadata
    name: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    coordinator_id: str
    peer_ids: List[str]
    coordinator_heartbeats_queue: str
    peer_command_queues: List[str]
