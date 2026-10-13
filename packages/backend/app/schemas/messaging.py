from pydantic import BaseModel


class SendMessageRequest(BaseModel):
    text: str


class SendFileRequest(BaseModel):
    file_url: str
    caption: str | None = None
    file_name: str | None = None


class ReplyMessageRequest(BaseModel):
    telegram_message_id: int
    text: str


class SendMessageResponse(BaseModel):
    telegram_message_id: int
    chat_id: str
    text: str | None = None


class SendFileResponse(BaseModel):
    telegram_message_id: int
    chat_id: str
    file_name: str | None = None


class SavedFileResponse(SendFileResponse):
    file_size: int
    sha256: str
