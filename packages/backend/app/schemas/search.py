import enum
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.chat import ChatType


class SearchMode(str, enum.Enum):
    HYBRID = "hybrid"
    EXACT = "exact"


class SearchRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.HYBRID
    chat_ids: list[UUID] | None = None
    chat_types: list[ChatType] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = None


class SearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    chat_id: UUID
    chat_title: str
    chat_type: ChatType | None = None
    chat_telegram_id: int | None = None
    chat_username: str | None = None
    telegram_message_id: int
    text: str | None
    sender_name: str | None
    is_outgoing: bool
    sent_at: datetime
    similarity: float
    has_media: bool = False
    media_type: str | None = None
    content_summary: str | None = None
    content_preview: str | None = None
    media_processing_status: str | None = None
    media_file_name: str | None = None
    media_mime_type: str | None = None
    media_file_size: int | None = None
    visible_urls: list[str] = Field(default_factory=list)
    hidden_urls: list[str] = Field(default_factory=list)
    deleted_at: datetime | None = None
    transcribed_at: datetime | None = None
    telegram_chat_url: str | None = None
    telegram_message_url: str | None = None
    media_download_url: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str
    total: int
    has_more: bool = False
    next_cursor: str | None = None
