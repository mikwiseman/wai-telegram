from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.limiter import limiter
from app.core.security import create_onboarding_token
from app.schemas.auth import TokenResponse
from app.services.onboarding import accept_invitation

router = APIRouter()


class AcceptInvitationRequest(BaseModel):
    token: str = Field(min_length=32, max_length=128)


@router.post("/accept", response_model=TokenResponse)
@limiter.limit("10/minute")
async def accept(
    request: Request,
    response: Response,
    body: AcceptInvitationRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    user_id = await accept_invitation(db, body.token)
    if user_id is None:
        raise HTTPException(401, "Ссылка уже использована или срок её действия истёк.")
    await db.commit()
    response.headers["Cache-Control"] = "no-store"
    return TokenResponse(
        access_token=create_onboarding_token(str(user_id)),
        refresh_token="",
    )
