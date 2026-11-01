#!/bin/bash
set -euo pipefail

readonly ENV_FILE="/opt/wai-telegram/.env.production"
readonly DATABASE_CONTAINER="wai-telegram-db"
readonly BOT_API_COMMIT="adfd7f6a8e990272851777eeb3ae0def4216f161"

usage() {
    echo "Usage: $0 --mode initial-cutover|standard --media-mode full|deferred" >&2
    exit 2
}

mode=""
media_mode=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode) mode="${2:-}"; shift 2 ;;
        --media-mode) media_mode="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[ "$mode" = "initial-cutover" ] || [ "$mode" = "standard" ] || usage
[ "$media_mode" = "full" ] || [ "$media_mode" = "deferred" ] || usage
[ "$(id -u)" -eq 0 ] || { echo "Run as root" >&2; exit 1; }
[ -r "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
source "$ENV_FILE"
set +a
: "${OWNER_USER_ID:?OWNER_USER_ID is required}"
: "${MEDIA_PIPELINE_ENABLED:?MEDIA_PIPELINE_ENABLED is required}"
: "${TELEGRAM_BOT_API_BASE_URL:?TELEGRAM_BOT_API_BASE_URL is required}"
case "$OWNER_USER_ID" in
    ????????-????-????-????-????????????) ;;
    *) echo "OWNER_USER_ID must be a UUID" >&2; exit 1 ;;
esac

command -v docker >/dev/null || { echo "Missing required command: docker" >&2; exit 1; }
docker exec "$DATABASE_CONTAINER" pg_isready -U telegram -d telegram_ai >/dev/null

cpu_count=$(nproc)
memory_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
media_bytes=0
if [ "$media_mode" = "full" ]; then
    [ "$MEDIA_PIPELINE_ENABLED" = "true" ] || {
        echo "Full media mode requires MEDIA_PIPELINE_ENABLED=true" >&2
        exit 1
    }
    # The local Bot API is what lifts the cloud API's 20MB limit, and only the
    # live-chat path uses it. A deployment that has turned that path off - the
    # archive downloads over MTProto - has no reason to run the server.
    if [ "${BOT_MEDIA_DOWNLOADS_ENABLED:-true}" != "false" ]; then
        case "$TELEGRAM_BOT_API_BASE_URL" in
            http://127.0.0.1:*|http://localhost:*|http://\[::1\]:*) ;;
            *) echo "Full media mode requires the local Bot API URL" >&2; exit 1 ;;
        esac
    fi
    # Storage and CPU requirements follow what is actually transcribed. Voice notes
    # and video notes are a few megabytes each and are deleted once their text is
    # extracted, so nothing accumulates and nothing durable needs backing up. Video,
    # audio and documents are the opposite: they are why the 500 GB volume, the
    # larger box and the off-host restic backups exist.
    speech_only=false
    case ",${MEDIA_TRANSCRIPTION_TYPES:-voice,video_note,audio,video}," in
        *,audio,*|*,video,*) ;;
        *) speech_only=true ;;
    esac

    if [ "$speech_only" = true ]; then
        media_free=$(df --output=avail -B1 "${MEDIA_ROOT:-/srv/wai-telegram-media}" | tail -n 1)
        [ "$cpu_count" -ge 2 ] && [ "$memory_kib" -ge 3500000 ] \
            && [ "$media_free" -ge 5000000000 ] || {
            echo "Speech-only media mode needs 2 CPU, 3.5 GB RAM, and 5 GB free on $MEDIA_ROOT" >&2
            exit 1
        }
        required_files=(/etc/wai-telegram/auth-backup-passphrase)
    else
        [ "$cpu_count" -ge 4 ] && [ "$memory_kib" -ge 7500000 ] || {
            echo "Server must be upgraded to at least 4 CPU / 8 GB RAM" >&2
            exit 1
        }
        mountpoint -q /srv/wai-telegram-media || {
            echo "A mounted media volume is required" >&2
            exit 1
        }
        media_bytes=$(df --output=size -B1 /srv/wai-telegram-media | tail -n 1)
        [ "$media_bytes" -ge 450000000000 ] || {
            echo "A mounted 500 GB-class media volume is required" >&2
            exit 1
        }
        required_files=(
            /etc/wai-telegram/restic.env
            /etc/wai-telegram/restic-password
            /etc/wai-telegram/auth-backup-passphrase
        )
    fi
else
    [ "$MEDIA_PIPELINE_ENABLED" = "false" ] || {
        echo "Deferred media mode requires MEDIA_PIPELINE_ENABLED=false" >&2
        exit 1
    }
    [ "$TELEGRAM_BOT_API_BASE_URL" = "https://api.telegram.org" ] || {
        echo "Deferred media mode requires the explicit cloud Bot API URL" >&2
        exit 1
    }
    free_bytes=$(df --output=avail -B1 / | tail -n 1)
    [ "$cpu_count" -ge 2 ] && [ "$memory_kib" -ge 3500000 ] \
        && [ "$free_bytes" -ge 8000000000 ] || {
        echo "Deferred release needs 2 CPU, 3.5 GB RAM, and 8 GB free" >&2
        exit 1
    }
    required_files=(/etc/wai-telegram/auth-backup-passphrase)
fi
for required_file in "${required_files[@]}"; do
    [ -r "$required_file" ] || {
        echo "Missing required root configuration: $required_file" >&2
        exit 1
    }
    [ "$(stat -c '%u:%a' "$required_file")" = "0:600" ] || {
        echo "Root configuration must be owned by root with mode 0600: $required_file" >&2
        exit 1
    }
done

psql_query() {
    printf '%s\n' "$1" | docker exec -i "$DATABASE_CONTAINER" psql \
        --username telegram \
        --dbname telegram_ai \
        --no-align \
        --tuples-only \
        --set ON_ERROR_STOP=1 \
        --set owner="$OWNER_USER_ID"
}

if [ "$mode" = "initial-cutover" ]; then
    invited=$(psql_query "SELECT count(*) FROM users u WHERE to_jsonb(u)->>'invited_by_id' IS NOT NULL;")
    [ "$invited" = "0" ] || {
        echo "Initial single-owner cutover is disabled after accounts have been invited" >&2
        exit 1
    }
    evidence=$(psql_query "
        WITH expected AS (SELECT :'owner'::uuid AS id),
        session_users AS (
          SELECT coalesce(array_agg(DISTINCT user_id ORDER BY user_id), ARRAY[]::uuid[]) AS ids
          FROM telegram_sessions WHERE is_active IS TRUE
        ),
        key_users AS (
          SELECT coalesce(array_agg(DISTINCT user_id ORDER BY user_id), ARRAY[]::uuid[]) AS ids
          FROM api_keys
          WHERE is_active IS TRUE AND last_used_at >= now() - interval '60 minutes'
        ),
        volumes AS (
          SELECT c.user_id, count(DISTINCT c.id) AS chats, count(m.id) AS messages
          FROM telegram_chats c LEFT JOIN telegram_messages m ON m.chat_id = c.id
          GROUP BY c.user_id
        ),
        top_users AS (
          SELECT coalesce(array_agg(user_id ORDER BY user_id), ARRAY[]::uuid[]) AS ids
          FROM volumes WHERE messages = (SELECT max(messages) FROM volumes) AND messages > 0
        )
        SELECT
          (session_users.ids = ARRAY[expected.id])::int || '|' ||
          (key_users.ids = ARRAY[expected.id])::int || '|' ||
          (top_users.ids = ARRAY[expected.id])::int || '|' ||
          coalesce((SELECT chats FROM volumes WHERE user_id = expected.id), 0) || '|' ||
          coalesce((SELECT messages FROM volumes WHERE user_id = expected.id), 0) || '|' ||
          (SELECT count(*) FROM users)
        FROM expected, session_users, key_users, top_users;")
    IFS='|' read -r session_ok key_ok volume_ok owner_chats owner_messages total_users <<<"$evidence"
    [ "$session_ok$key_ok$volume_ok" = "111" ] || {
        echo "Owner evidence is ambiguous; no production mutation is allowed" >&2
        exit 1
    }
    echo "initial-cutover preflight: owner_chats=$owner_chats owner_messages=$owner_messages total_users=$total_users"
else
    state=$(psql_query "
        WITH expected AS (SELECT :'owner'::uuid AS id)
        SELECT
          (SELECT count(*) FROM users u, expected WHERE u.is_active IS TRUE AND u.id <> expected.id AND
            (coalesce(to_jsonb(u)->>'invited_by_id', '') <> expected.id::text OR to_jsonb(u)->>'expected_telegram_user_id' IS NULL)) || '|' ||
          (SELECT count(*) FROM users, expected WHERE users.id = expected.id AND users.is_active IS TRUE) || '|' ||
          (SELECT count(*) FROM api_keys k JOIN users u ON u.id = k.user_id WHERE u.is_active IS FALSE AND k.is_active IS TRUE) || '|' ||
          (SELECT count(*) FROM telegram_sessions s JOIN users u ON u.id = s.user_id WHERE u.is_active IS FALSE AND (s.is_active IS TRUE OR s.session_string <> ''))
        FROM expected;")
    IFS='|' read -r active_users owner_active archive_keys archive_sessions <<<"$state"
    [ "$active_users|$owner_active|$archive_keys|$archive_sessions" = "0|1|0|0" ] || {
        echo "Owner/invited-account production invariants failed" >&2
        exit 1
    }
    echo "standard preflight: owner and explicitly invited accounts only; archive credentials revoked"
fi

echo "read-only infrastructure snapshot: media_mode=$media_mode cpu=$cpu_count memory_kib=$memory_kib media_bytes=$media_bytes bot_api_commit=$BOT_API_COMMIT"
