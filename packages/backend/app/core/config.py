from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "Telegram AI Message Manager"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://telegram:telegram_dev@localhost:5432/telegram_ai"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")

    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production")
    encryption_key: str = Field(default="")  # Fernet key for session encryption
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    algorithm: str = "HS256"
    owner_user_id: UUID | None = None
    correspondence_webhook_url: str = ""
    correspondence_webhook_token: str = Field(default="", repr=False)

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Telegram
    telegram_api_id: int = Field(default=0)
    telegram_api_hash: str = Field(default="")
    telegram_bot_token: str = Field(default="")
    telegram_bot_api_base_url: str = "http://127.0.0.1:8081"

    # OpenAI generation, vision, and embeddings
    openai_api_key: str = Field(default="")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 100
    generation_model: str = "gpt-5.6-luna"
    fast_generation_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] = "none"
    quality_generation_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] = "low"

    # Sentry (error tracking)
    sentry_dsn: str = Field(default="")
    sentry_release: str = Field(default="")
    sentry_traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    sentry_profiles_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    sentry_enable_logs: bool = True
    sentry_debug: bool = False

    # Cloudflare (site deployment)
    cloudflare_api_token: str = Field(default="")
    cloudflare_account_id: str = Field(default="")

    # DeepGram (voice transcription)
    deepgram_api_key: str = Field(default="")
    deepgram_model: str = "nova-3"
    deepgram_language: str = "multi"
    deepgram_timeout_seconds: float = 600.0
    media_transcription_chunk_seconds: int = 600

    # Background media processing
    media_summary_model: str = "gpt-5.6-luna"
    media_summary_max_output_tokens: int = 500
    media_image_max_output_tokens: int = 1600
    media_summary_chunk_chars: int = 120_000
    media_embedding_chunk_chars: int = 6_000
    media_embedding_chunk_overlap_chars: int = 600
    document_extraction_timeout_seconds: float = 180.0
    pdf_ocr_min_text_chars: int = Field(default=200, ge=0)
    pdf_ocr_batch_pages: int = Field(default=8, ge=1, le=50)
    pdf_ocr_dpi: int = Field(default=160, ge=72, le=300)
    video_frame_interval_seconds: int = Field(default=30, ge=1)
    video_scene_threshold: float = Field(default=0.3, gt=0.0, lt=1.0)
    video_frame_analysis_batch: int = Field(default=4, ge=1, le=16)
    media_pipeline_enabled: bool = True
    # Comma-separated subset of voice,video_note,audio,video. Unset means all four.
    media_transcription_types: str | None = None
    # Whether the bot fetches media a user attaches in a live chat. That path, and
    # only that path, goes through the Bot API (app/services/agent/media_processor.py)
    # and needs the local server to exceed the cloud API's 20MB file limit. The
    # historical archive pipeline downloads over MTProto via Telethon and never
    # touches it, so it must not be gated on the same requirement. Leave enabled
    # unless the deployment deliberately serves archive search only.
    bot_media_downloads_enabled: bool = True
    media_root: Path = Path("/srv/wai-telegram-media")
    media_internal_uri_prefix: str = "/_protected_media"
    media_download_stall_timeout_seconds: float = Field(default=120.0, gt=0.0)
    media_download_chunk_bytes: int = Field(default=512 * 1024, ge=64 * 1024)
    media_progress_checkpoint_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    media_lock_ttl_seconds: int = Field(default=300, ge=30)
    media_dispatch_target_depth: int = 20
    media_queue_stale_minutes: int = 360
    media_processing_stale_minutes: int = 120

    # Sync settings
    sync_batch_size: int = 100
    sync_delay_seconds: float = 1.0
    sync_delay_jitter: float = 0.5
    sync_progressive_delay_interval: int = 5  # Increase delay every N batches
    sync_progressive_delay_step: float = 0.5  # Add this much per interval
    # Zero means all Telegram dialogs. A finite default leaves older unread
    # counts stale and makes unread-only scans incomplete.
    sync_dialog_limit: int = 0
    flood_wait_multiplier: float = 1.2

    # Telegram client settings (anti-ban)
    telegram_device_model: str = "MacBook Pro"
    telegram_system_version: str = "macOS 14.5"
    telegram_app_version: str = "10.8.1"
    telegram_flood_sleep_threshold: int = 120

    # Rate budget tracking
    rate_budget_hourly: int = 200
    rate_budget_daily: int = 2000

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Ensure critical secrets are set outside local development."""
        if self.environment in {"staging", "production"}:
            if self.secret_key == "dev-secret-key-change-in-production":
                raise ValueError("SECRET_KEY must be set in staging/production")
            if not self.encryption_key:
                raise ValueError("ENCRYPTION_KEY must be set in staging/production")
            if not self.telegram_api_id or not self.telegram_api_hash:
                raise ValueError("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set")
            if self.environment == "production" and self.owner_user_id is None:
                raise ValueError("OWNER_USER_ID must be set in production")
            if (
                self.environment == "production"
                and self.media_pipeline_enabled
                and self.bot_media_downloads_enabled
            ):
                bot_api_host = urlparse(self.telegram_bot_api_base_url).hostname
                if bot_api_host not in {"127.0.0.1", "localhost", "::1"}:
                    raise ValueError(
                        "TELEGRAM_BOT_API_BASE_URL must point to the local Bot API"
                    )
        return self

    @computed_field
    @property
    def async_database_url(self) -> str:
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
