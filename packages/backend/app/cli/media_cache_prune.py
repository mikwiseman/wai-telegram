"""Drop cached originals once their text is stored, and say so in the database.

The media volume is far smaller than the archive it serves, so an original is
worth keeping only until its text has been extracted. Deleting the file alone is
not enough: media_objects still carries relative_path and sha256, which is the
only thing the tools read to decide a file is downloadable. Left that way the
row promises bytes that are gone - download_media hands out a signed URL, the
volume 404s behind it, and prepare_media refuses to refetch because the status
still says ready.

So the row goes with the file. The message keeps its transcript and summary, the
listing reports the file as not_prepared rather than ready, and asking for it
fetches it from Telegram again.

Originals, partial downloads, and extraction artifacts can outlive their records.
Sweep those bytes too, while preserving every current cache key (including
downloads that have no relative_path yet) and recently written files.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.database import get_db_context
from app.core.observability import configure_logging, init_observability
from app.models.media import MediaObject
from app.models.message import MediaProcessingStatus, TelegramMessage

logger = logging.getLogger(__name__)

# Long enough that a caller who just asked for a file still finds it there when
# the download link is used, short enough that the volume never fills.
DEFAULT_RETENTION_MINUTES = 120

# Text has to be safely stored before the bytes go, so only settled work is
# pruned. Anything still moving keeps its file until it settles.
_SETTLED = (
    MediaProcessingStatus.READY,
    MediaProcessingStatus.SKIPPED,
    MediaProcessingStatus.FAILED,
)


async def prune_media_cache(retention_minutes: int = DEFAULT_RETENTION_MINUTES) -> dict:
    settings = get_settings()
    root = Path(settings.media_root)
    cutoff = datetime.now(UTC) - timedelta(minutes=retention_minutes)
    deleted_files = 0
    freed_bytes = 0
    cleared_rows = 0
    missing_rows = 0

    async with get_db_context() as db:
        rows = (
            await db.execute(
                select(
                    MediaObject.id,
                    MediaObject.relative_path,
                    MediaObject.size_bytes,
                    MediaObject.fetched_at,
                    TelegramMessage.media_processing_status,
                )
                .join(TelegramMessage, TelegramMessage.id == MediaObject.message_id)
                .where(MediaObject.relative_path.isnot(None))
            )
        ).all()

        expired: list = []
        for row in rows:
            path = root / row.relative_path
            if not path.exists():
                # The file is already gone; the row is the only thing still
                # claiming otherwise.
                missing_rows += 1
                expired.append(row.id)
                continue
            if row.media_processing_status not in _SETTLED:
                continue
            fetched = row.fetched_at
            if fetched is not None and fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=UTC)
            if fetched is not None and fetched > cutoff:
                continue
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                logger.exception("Could not remove cached original %s", path)
                continue
            deleted_files += 1
            freed_bytes += size
            expired.append(row.id)

        for index in range(0, len(expired), 500):
            batch = expired[index : index + 500]
            await db.execute(delete(MediaObject).where(MediaObject.id.in_(batch)))
            cleared_rows += len(batch)
        await db.commit()

        # Read keys after the row cleanup. A download gets its key before it
        # gets a relative_path, so querying only non-null paths would remove
        # resumable work. Keep all files belonging to a current key.
        current = (
            await db.execute(select(MediaObject.cache_key, MediaObject.relative_path))
        ).all()
        orphaned_files, orphaned_bytes = _remove_orphaned_files(
            root,
            cutoff,
            cache_keys={row.cache_key for row in current},
            relative_paths={row.relative_path for row in current if row.relative_path},
        )
        deleted_files += orphaned_files
        freed_bytes += orphaned_bytes

    _remove_empty_dirs(root)
    result = {
        "deleted_files": deleted_files,
        "freed_bytes": freed_bytes,
        "cleared_rows": cleared_rows,
        "rows_already_missing_a_file": missing_rows,
        "deleted_orphan_files": orphaned_files,
        "retention_minutes": retention_minutes,
    }
    logger.info(
        "Pruned %s cached files (%s bytes, %s orphaned), cleared %s rows (%s had no file)",
        deleted_files,
        freed_bytes,
        orphaned_files,
        cleared_rows,
        missing_rows,
    )
    return result


def _remove_orphaned_files(
    root: Path,
    cutoff: datetime,
    *,
    cache_keys: set[str],
    relative_paths: set[str],
) -> tuple[int, int]:
    if not root.exists():
        return 0, 0

    deleted_files = 0
    freed_bytes = 0
    hex_digits = frozenset("0123456789abcdef")
    for prefix in root.iterdir():
        if (
            len(prefix.name) != 2
            or not set(prefix.name) <= hex_digits
            or prefix.is_symlink()
            or not prefix.is_dir()
        ):
            continue
        for directory in prefix.iterdir():
            key = directory.name
            if (
                len(key) != 64
                or not set(key) <= hex_digits
                or not key.startswith(prefix.name)
                or key in cache_keys
                or directory.is_symlink()
                or not directory.is_dir()
            ):
                continue
            for path in directory.rglob("*"):
                if (
                    not _is_cache_artifact(path.relative_to(directory))
                    or path.is_symlink()
                    or not path.is_file()
                    or str(path.relative_to(root)) in relative_paths
                ):
                    continue
                try:
                    metadata = path.stat()
                    if metadata.st_mtime > cutoff.timestamp():
                        continue
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    logger.exception("Could not remove orphaned cache file %s", path)
                    continue
                deleted_files += 1
                freed_bytes += metadata.st_size
    return deleted_files, freed_bytes


def _is_cache_artifact(relative_path: Path) -> bool:
    if len(relative_path.parts) == 1:
        return relative_path.name.startswith("original.") or any(
            relative_path.match(pattern)
            for pattern in (
                "chunk-*.ogg",
                "transcript-*.json",
                "transcript-*.json.part",
                "extracted-content.md",
                "visual-timeline.md",
                "visual-timeline.md.part",
                "pdf-ocr.md",
                "pdf-ocr.md.part",
            )
        )
    return (
        len(relative_path.parts) == 2
        and relative_path.parts[0] in {"visual-timeline", "pdf-ocr"}
        and relative_path.suffix in {".jpg", ".png"}
    )


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_symlink() and path.is_dir() and not any(path.iterdir()):
            try:
                path.rmdir()
            except OSError:
                pass


def main() -> None:
    settings = get_settings()
    configure_logging()
    init_observability(settings, "wai-telegram-media-cache-prune")
    asyncio.run(prune_media_cache())


if __name__ == "__main__":
    main()
