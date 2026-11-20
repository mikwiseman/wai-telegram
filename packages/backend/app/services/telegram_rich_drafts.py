"""Native Telegram editor drafts with an inline photo or document."""

import mimetypes
from pathlib import Path

from telethon import functions, types, utils


async def read_peer_draft(client, peer):
    result = await client(
        functions.messages.GetPeerDialogsRequest([types.InputDialogPeer(peer)])
    )
    for dialog in result.dialogs:
        if isinstance(peer, types.InputPeerSelf) or utils.get_peer_id(
            dialog.peer
        ) == utils.get_peer_id(peer):
            return dialog.draft
    raise ValueError("Telegram did not return the requested dialog")


def draft_state(draft):
    """Compare editable contents, ignoring Telegram's rotating file references."""
    if draft is None:
        return None
    state = draft.to_dict()
    rich = state.get("rich_message")
    if rich:
        for key in ("photos", "documents"):
            rich[key] = [media["id"] for media in rich.get(key) or []]
    return state


def _plain_text(rich_text):
    if isinstance(rich_text, types.TextEmpty):
        return ""
    if isinstance(rich_text, types.TextConcat):
        return "".join(_plain_text(item) for item in rich_text.texts)
    inner = getattr(rich_text, "text", None)
    if isinstance(inner, str):
        return inner
    return _plain_text(inner) if inner is not None else ""


def rich_draft_content(rich):
    photos = {p.id for p in rich.photos if isinstance(p, types.Photo)}
    documents = {d.id: d for d in rich.documents if isinstance(d, types.Document)}
    paragraphs = []
    attachments = []
    for block in rich.blocks:
        if isinstance(block, types.PageBlockParagraph):
            paragraphs.append(_plain_text(block.text))
        elif isinstance(block, types.PageBlockPhoto) and block.photo_id in photos:
            attachments.append(
                {"kind": "photo", "telegram_file_id": str(block.photo_id)}
            )
        elif (
            isinstance(block, types.PageBlockDocument)
            and block.document_id in documents
        ):
            doc = documents[block.document_id]
            filename = next(
                (
                    a.file_name
                    for a in doc.attributes
                    if isinstance(a, types.DocumentAttributeFilename)
                ),
                None,
            )
            attachments.append(
                {
                    "kind": "document",
                    "telegram_file_id": str(doc.id),
                    "file_name": filename,
                    "size": doc.size,
                    "mime_type": doc.mime_type,
                }
            )
    return "\n\n".join(paragraphs), attachments


async def save_with_attachment(client, peer, text: str, path: Path, previous):
    with path.open("rb") as source:
        signature = source.read(8)
    photo = signature.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"))
    uploaded = await client.upload_file(path, file_name=path.name)
    if photo:
        media = types.InputMediaUploadedPhoto(uploaded)
    else:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        media = types.InputMediaUploadedDocument(
            uploaded,
            mime,
            [types.DocumentAttributeFilename(path.name)],
            force_file=True,
        )
    prepared = await client(
        functions.messages.UploadMediaRequest(peer=peer, media=media)
    )
    caption = types.PageCaption(types.TextEmpty(), types.TextEmpty())
    blocks = []
    if text:
        blocks.extend(
            types.PageBlockParagraph(types.TextPlain(p)) for p in text.split("\n\n")
        )
    if photo:
        if not isinstance(getattr(prepared, "photo", None), types.Photo):
            raise ValueError("Telegram did not prepare the uploaded photo")
        ref = utils.get_input_photo(prepared.photo)
        blocks.append(types.PageBlockPhoto(ref.id, caption))
        rich = types.InputRichMessage(blocks=blocks, photos=[ref])
    else:
        if not isinstance(getattr(prepared, "document", None), types.Document):
            raise ValueError("Telegram did not prepare the uploaded document")
        ref = utils.get_input_document(prepared.document)
        blocks.append(types.PageBlockDocument(ref.id, caption))
        rich = types.InputRichMessage(blocks=blocks, documents=[ref])

    if draft_state(await read_peer_draft(client, peer)) != draft_state(previous):
        raise ValueError(
            "The Telegram draft changed during the upload. Read it again before replacing it."
        )
    saved = await client(
        functions.messages.SaveDraftRequest(peer=peer, message=text, rich_message=rich)
    )
    if not saved:
        raise ValueError("Telegram did not confirm that the draft was saved")
    draft = await read_peer_draft(client, peer)
    returned_rich = getattr(draft, "rich_message", None)
    if returned_rich:
        actual_text, attachments = rich_draft_content(returned_rich)
        if actual_text == text and len(attachments) == 1:
            attachment = attachments[0]
            if attachment["telegram_file_id"] == str(ref.id):
                if photo and attachment["kind"] == "photo":
                    return attachment
                if (
                    attachment["kind"] == "document"
                    and attachment["file_name"] == path.name
                    and attachment["size"] == path.stat().st_size
                ):
                    return attachment
    raise ValueError("Telegram did not return the requested draft text and attachment")
