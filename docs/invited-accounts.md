# Invited Telegram accounts

The owner invites a Telegram user ID. The recipient opens one link, signs into that exact Telegram account, selects chats, and connects a personal MCP key. No separate WAI password is required.

## Create or renew a link

Run with the production environment loaded, from `packages/backend`:

```sh
uv run python -m app.cli.invite_user --telegram-user-id TELEGRAM_USER_ID
```

The private link expires after seven days and can be redeemed once. Its fragment is removed immediately. Redemption grants only a 30-minute Telegram-login token; archive access and key creation require successful Telegram authentication. Reissuing a link retains the existing account and chat choices.

## Indexing

- Invited accounts start with no chats selected. Selection loads all available history; the existing listener receives new messages.
- Message and file search use selected chats. Other chat metadata remains visible for selection and sending; selection controls indexing, not a separate access role.
- History checkpoints survive interruption. The existing maintenance task recovers lost dispatches and unfinished selections. Temporary failures retry after 15 minutes.
- Missing text embeddings retry each minute, up to ten batches of 100 per opted-in user. Media uses the existing extraction pipeline.
- History and embedding repair pause below 8 GB free on the database host. `/connect` and `get_data_status` show coverage.
- Deselecting stops indexing and removes the chat from search, without erasing its archive. Reselecting rescans with deduplication to recover messages received while disabled.

Migration 030 keeps existing chats enabled but does not opt the owner into mass backfill. Saving a selection explicitly enables that account's backfill. Old single-owner cutover commands refuse to run after any invitation exists.

## Deployment

Use CI / Deploy on main in `standard` mode. Migration 030 removes the single-active-account index; startup still requires the configured owner and rejects uninvited active accounts. Old code cannot serve two active accounts: a rollback must retain the new account model or restore the verified pre-deploy backup.

Tests cover bootstrap restrictions, wrong-account rejection, PostgreSQL account/chat search isolation, embedding retry, history continuation, MCP session separation and browser-cache separation. The browser smoke uses synthetic Telegram authentication. The recipient completes real Telegram login and their AI client's connection.
