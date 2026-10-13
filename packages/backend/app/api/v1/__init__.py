from fastapi import APIRouter

from app.api.v1 import (
    auth,
    bot_webhook,
    chats,
    digests,
    messages,
    post_drafts,
    search,
    settings,
    sync,
    telegram,
    tools,
)

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(telegram.router, prefix="/telegram", tags=["telegram"])
api_router.include_router(bot_webhook.router, prefix="/bot", tags=["bot"])
api_router.include_router(chats.router, prefix="/chats", tags=["chats"])
api_router.include_router(messages.router, prefix="/messages", tags=["messages"])
api_router.include_router(sync.router, prefix="/sync", tags=["sync"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(digests.router, prefix="/digests", tags=["digests"])
api_router.include_router(settings.router, prefix="/settings", tags=["settings"])
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])

api_router.include_router(
    post_drafts.router, prefix="/post-drafts", tags=["post-drafts"]
)
