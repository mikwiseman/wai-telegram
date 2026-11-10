#!/bin/bash
set -euo pipefail
umask 077

readonly BACKUP_ROOT="/opt/wai-telegram-backups"
readonly DATABASE_CONTAINER="wai-telegram-db"
readonly PASSPHRASE_FILE="/etc/wai-telegram/auth-backup-passphrase"

[ "$(id -u)" -eq 0 ] || { echo "Run as root" >&2; exit 1; }
[ -r "$PASSPHRASE_FILE" ] || { echo "Missing $PASSPHRASE_FILE" >&2; exit 1; }
[ -s "$PASSPHRASE_FILE" ] || { echo "Empty $PASSPHRASE_FILE" >&2; exit 1; }
[ "$(stat -c '%u:%a' "$PASSPHRASE_FILE")" = "0:600" ] || {
    echo "$PASSPHRASE_FILE must be owned by root with mode 0600" >&2
    exit 1
}

# The dump is written next to the previous one, which stays until the new one is verified.
# Refuse to start when it cannot fit: a full disk stops the database on this same volume.
mkdir -p "$BACKUP_ROOT"
previous_dump=$(ls -t "$BACKUP_ROOT"/auth-cutover-*/database.dump.gpg 2>/dev/null | head -n 1 || true)
if [ -n "$previous_dump" ]; then
    expected_bytes=$(( $(stat -c %s "$previous_dump") * 12 / 10 ))
else
    expected_bytes=$(docker exec "$DATABASE_CONTAINER" psql --username telegram --dbname telegram_ai \
        --tuples-only --no-align --command "select pg_database_size(current_database())")
fi
free_bytes=$(df --output=avail -B1 "$BACKUP_ROOT" | tail -n 1)
reserve_bytes=$((3 * 1024 * 1024 * 1024))
[ "$free_bytes" -ge $((expected_bytes + reserve_bytes)) ] || {
    echo "Not enough disk space for the backup: about $((expected_bytes / 1000000000)) GB needed plus 3 GB reserve, $((free_bytes / 1000000000)) GB free" >&2
    exit 1
}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
destination="$BACKUP_ROOT/auth-cutover-$timestamp"
incomplete_destination="$BACKUP_ROOT/.auth-cutover-$timestamp.incomplete"
mkdir -p "$incomplete_destination"
encrypted_dump="$incomplete_destination/database.dump.gpg"
restore_list="$incomplete_destination/database.restore-list.txt"
checksum_file="$incomplete_destination/SHA256SUMS"

cleanup() {
    rm -f -- "$encrypted_dump" "$restore_list" "$checksum_file"
    rmdir -- "$incomplete_destination" 2>/dev/null || true
}
trap cleanup EXIT

docker exec "$DATABASE_CONTAINER" pg_dump \
    --username telegram \
    --dbname telegram_ai \
    --format custom \
    --no-owner \
    --no-acl \
    | gpg --batch --yes --pinentry-mode loopback --cipher-algo AES256 --force-mdc \
    --s2k-digest-algo SHA512 --s2k-count 65011712 \
    --passphrase-file "$PASSPHRASE_FILE" \
    --symmetric --output "$encrypted_dump"

# Consume the entire authenticated stream first. pg_restore --list only needs the
# custom archive TOC and may close stdin early, so it cannot be the integrity check.
gpg --batch --yes --pinentry-mode loopback \
    --passphrase-file "$PASSPHRASE_FILE" \
    --decrypt "$encrypted_dump" >/dev/null

set +o pipefail
gpg --batch --yes --pinentry-mode loopback \
    --passphrase-file "$PASSPHRASE_FILE" \
    --decrypt "$encrypted_dump" \
    | docker exec -i "$DATABASE_CONTAINER" pg_restore --list > "$restore_list"
restore_status="${PIPESTATUS[1]}"
set -o pipefail
[ "$restore_status" -eq 0 ] || {
    echo "pg_restore could not read the encrypted backup" >&2
    exit 1
}
[ -s "$restore_list" ] || { echo "Backup verification produced no manifest" >&2; exit 1; }

(
    cd "$incomplete_destination"
    sha256sum database.dump.gpg > SHA256SUMS
    sha256sum --check SHA256SUMS
)
chmod 0600 "$encrypted_dump" "$checksum_file" "$restore_list"
mv "$incomplete_destination" "$destination"
trap - EXIT

# Every deploy writes a snapshot large enough that retaining two leaves less
# free space than the next deploy needs. Keep the newest verified snapshot;
# rotation happens only after the fresh one above passed every check, so there
# is always at least one restorable database copy.
ls -dt "$BACKUP_ROOT"/auth-cutover-* | tail -n +2 | xargs -r rm -rf

echo "$destination"
