from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMediaInfo:
    def test_voice_attribute_wins_over_generic_document_filename(self):
        from app.services.media_content_service import get_media_info

        message = SimpleNamespace(
            media=SimpleNamespace(
                document=SimpleNamespace(
                    mime_type="audio/ogg",
                    size=1234,
                    attributes=[
                        SimpleNamespace(voice=True, duration=17),
                        SimpleNamespace(file_name="voice.ogg"),
                    ],
                )
            )
        )

        info = get_media_info(message)

        assert info.media_type == "voice"
        assert info.file_name == "voice.ogg"
        assert info.mime_type == "audio/ogg"
        assert info.file_size == 1234
        assert info.duration_seconds == 17

    @pytest.mark.parametrize(
        ("mime_type", "expected"),
        [
            ("audio/mpeg", "audio"),
            ("video/mp4", "video"),
            ("image/png", "photo"),
            ("application/pdf", "document"),
        ],
    )
    def test_document_category_uses_mime_type(self, mime_type, expected):
        from app.services.media_content_service import get_media_info

        message = SimpleNamespace(
            media=SimpleNamespace(
                document=SimpleNamespace(
                    mime_type=mime_type,
                    size=99,
                    attributes=[SimpleNamespace(file_name="sample.bin")],
                )
            )
        )

        assert get_media_info(message).media_type == expected


class TestChunkText:
    def test_chunks_long_content_without_losing_text(self):
        from app.services.media_content_service import chunk_text

        text = "\n\n".join(f"Paragraph {index}: " + ("x" * 80) for index in range(40))
        chunks = chunk_text(text, max_chars=500, overlap_chars=0)

        assert len(chunks) > 1
        assert all(len(chunk) <= 500 for chunk in chunks)
        assert "\n\n".join(chunks) == text


class TestDeepgramTranscription:
    async def test_returns_diarized_utterances_with_timecodes(self, tmp_path):
        from app.services.media_content_service import transcribe_media_file_detailed

        media_path = tmp_path / "voice.ogg"
        media_path.write_bytes(b"audio bytes")

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": {
                "channels": [
                    {
                        "detected_language": "ru",
                        "alternatives": [{"transcript": "Привет, мир."}],
                    }
                ],
                "utterances": [
                    {
                        "start": 1.25,
                        "end": 2.75,
                        "speaker": 0,
                        "confidence": 0.97,
                        "transcript": "Привет,",
                    },
                    {
                        "start": 2.8,
                        "end": 3.5,
                        "speaker": 1,
                        "confidence": 0.91,
                        "transcript": "мир.",
                    },
                ],
            }
        }
        client = AsyncMock()
        client.post.return_value = response

        with (
            patch("app.services.media_content_service.settings") as mock_settings,
            patch(
                "app.services.media_content_service.httpx.AsyncClient"
            ) as client_class,
        ):
            mock_settings.deepgram_api_key = "test-key"
            mock_settings.deepgram_model = "nova-3"
            mock_settings.deepgram_language = "multi"
            client_class.return_value.__aenter__.return_value = client

            result = await transcribe_media_file_detailed(media_path, "audio/ogg")

        assert result.text == "Привет, мир."
        assert [(item.start_ms, item.end_ms) for item in result.segments] == [
            (1250, 2750),
            (2800, 3500),
        ]
        assert [item.speaker for item in result.segments] == ["0", "1"]
        assert [item.language for item in result.segments] == ["ru", "ru"]
        request = client.post.await_args
        assert request.kwargs["params"] == {
            "model": "nova-3",
            "language": "multi",
            "smart_format": "true",
            "utterances": "true",
            "diarize": "true",
        }
        assert "detect_language" not in request.kwargs["params"]
        assert "punctuate" not in request.kwargs["params"]
        assert request.kwargs["headers"]["Content-Type"] == "audio/ogg"

    async def test_plain_transcript_wrapper_preserves_existing_contract(self, tmp_path):
        from app.services.media_content_service import (
            TranscriptionResult,
            transcribe_media_file,
        )

        path = tmp_path / "audio.ogg"
        path.write_bytes(b"audio")
        with patch(
            "app.services.media_content_service.transcribe_media_file_detailed",
            new_callable=AsyncMock,
            return_value=TranscriptionResult(text="Готово.", segments=()),
        ):
            assert await transcribe_media_file(path, "audio/ogg") == "Готово."

    async def test_missing_deepgram_key_is_explicit_failure(self, tmp_path):
        from app.services.media_content_service import (
            MediaProcessingConfigurationError,
            transcribe_media_file,
        )

        media_path = tmp_path / "voice.ogg"
        media_path.write_bytes(b"audio bytes")

        with patch("app.services.media_content_service.settings") as mock_settings:
            mock_settings.deepgram_api_key = ""
            with pytest.raises(
                MediaProcessingConfigurationError,
                match="DEEPGRAM_API_KEY",
            ):
                await transcribe_media_file(media_path, "audio/ogg")

    async def test_empty_provider_transcript_is_failure(self, tmp_path):
        from app.services.media_content_service import (
            MediaNoSpeechError,
            transcribe_media_file,
        )

        media_path = tmp_path / "voice.ogg"
        media_path.write_bytes(b"audio bytes")

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": {"channels": []}}
        client = AsyncMock()
        client.post.return_value = response

        with (
            patch("app.services.media_content_service.settings") as mock_settings,
            patch(
                "app.services.media_content_service.httpx.AsyncClient"
            ) as client_class,
        ):
            mock_settings.deepgram_api_key = "test-key"
            mock_settings.deepgram_model = "nova-3"
            mock_settings.deepgram_language = "multi"
            client_class.return_value.__aenter__.return_value = client

            with pytest.raises(
                MediaNoSpeechError,
                match="no speech",
            ):
                await transcribe_media_file(media_path, "audio/ogg")


class TestOpenAISummaries:
    async def test_summary_uses_available_cost_optimized_gpt56_model(self):
        from app.services.media_content_service import ContentSummary, summarize_text

        parsed = ContentSummary(summary="Краткое содержание.")
        output_item = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", parsed=parsed)],
        )
        response = SimpleNamespace(output=[output_item])
        client = SimpleNamespace(
            responses=SimpleNamespace(parse=AsyncMock(return_value=response))
        )

        with (
            patch("app.services.media_content_service.settings") as mock_settings,
            patch(
                "app.services.media_content_service.get_openai_client",
                new_callable=AsyncMock,
                return_value=client,
            ),
        ):
            mock_settings.media_summary_model = "gpt-5.6-luna"
            mock_settings.media_summary_chunk_chars = 120_000
            result = await summarize_text("Полный текст")

        assert result == "Краткое содержание."
        request = client.responses.parse.await_args.kwargs
        assert request["model"] == "gpt-5.6-luna"
        assert request["reasoning"] == {"effort": "none"}
        assert request["store"] is False
        assert request["text_format"] is ContentSummary

    async def test_image_analysis_uses_low_detail_and_structured_output(self, tmp_path):
        from app.services.media_content_service import ImageAnalysis, analyze_image

        image_path = tmp_path / "photo.jpg"
        image_path.write_bytes(b"jpeg")
        parsed = ImageAnalysis(
            visible_text="WAI",
            summary="Логотип WAI на светлом фоне.",
        )
        output_item = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", parsed=parsed)],
        )
        response = SimpleNamespace(output=[output_item])
        client = SimpleNamespace(
            responses=SimpleNamespace(parse=AsyncMock(return_value=response)),
            files=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(id="file-image")),
                delete=AsyncMock(),
            ),
        )

        with (
            patch("app.services.media_content_service.settings") as mock_settings,
            patch(
                "app.services.media_content_service.get_openai_client",
                new_callable=AsyncMock,
                return_value=client,
            ),
        ):
            mock_settings.media_summary_model = "gpt-5.6-luna"
            mock_settings.media_image_max_output_tokens = 1600
            result = await analyze_image(image_path, "image/jpeg")

        assert result == parsed
        request = client.responses.parse.await_args.kwargs
        assert request["input"][0]["role"] == "developer"
        image_input = request["input"][1]["content"][1]
        assert image_input["type"] == "input_image"
        assert image_input["file_id"] == "file-image"
        assert "image_url" not in image_input
        assert image_input["detail"] == "high"
        assert request["reasoning"] == {"effort": "none"}
        assert request["max_output_tokens"] == 1600
        upload = client.files.create.await_args
        assert upload.kwargs["purpose"] == "user_data"
        assert upload.kwargs["expires_after"]["seconds"] == 3600
        client.files.delete.assert_awaited_once_with("file-image")

    async def test_invalid_structured_image_response_is_provider_error(self, tmp_path):
        from pydantic import ValidationError

        from app.services.media_content_service import (
            ImageAnalysis,
            MediaProviderResponseError,
            analyze_image,
        )

        image_path = tmp_path / "photo.jpg"
        image_path.write_bytes(b"jpeg")
        try:
            ImageAnalysis.model_validate_json('{"visible_text":"truncated')
        except ValidationError as validation_error:
            error = validation_error

        client = SimpleNamespace(
            responses=SimpleNamespace(parse=AsyncMock(side_effect=error)),
            files=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(id="file-image")),
                delete=AsyncMock(),
            ),
        )
        with (
            patch("app.services.media_content_service.settings") as mock_settings,
            patch(
                "app.services.media_content_service.get_openai_client",
                new_callable=AsyncMock,
                return_value=client,
            ),
        ):
            mock_settings.media_summary_model = "gpt-5.6-luna"
            mock_settings.media_image_max_output_tokens = 1600
            with pytest.raises(
                MediaProviderResponseError,
                match="invalid structured image",
            ):
                await analyze_image(image_path, "image/jpeg")


class TestDocumentExtraction:
    async def test_plain_text_document_is_extracted_as_searchable_text(self, tmp_path):
        from app.services.media_content_service import extract_document_text

        source = tmp_path / "notes.txt"
        source.write_text("# Заголовок\n\nПолный текст.", encoding="utf-8")

        result = await extract_document_text(source)

        assert "# Заголовок" in result
        assert "Полный текст." in result

    async def test_unknown_binary_format_is_reported_explicitly(self, tmp_path):
        from app.services.media_content_service import (
            MediaUnsupportedFormatError,
            extract_document_text,
        )

        source = tmp_path / "firmware.bin"
        source.write_bytes(bytes(range(32)))

        with pytest.raises(MediaUnsupportedFormatError, match="not supported"):
            await extract_document_text(source)

    async def test_scanned_pdf_ocr_covers_every_page_in_batches(self, tmp_path):
        from app.services.media_content_service import (
            ImageAnalysis,
            extract_scanned_pdf_text,
        )

        source = tmp_path / "scan.pdf"
        source.write_bytes(b"pdf")

        async def render_pages(*args, **_kwargs):
            first = int(args[args.index("-f") + 1])
            last = int(args[args.index("-l") + 1])
            prefix = tmp_path / "pdf-ocr" / f"batch-{first:06d}"
            prefix.parent.mkdir(exist_ok=True)
            for page in range(first, last + 1):
                (prefix.parent / f"{prefix.name}-{page:06d}.jpg").write_bytes(b"jpg")
            return 0, ""

        async def analyze(path, _mime):
            return ImageAnalysis(
                visible_text=f"Text from {path.name}",
                summary="Scanned page",
            )

        with (
            patch(
                "app.services.media_content_service.shutil.which",
                return_value="pdftoppm",
            ),
            patch(
                "app.services.media_content_service._pdf_page_count",
                new_callable=AsyncMock,
                return_value=10,
            ),
            patch(
                "app.services.media_content_service._run_process",
                new_callable=AsyncMock,
                side_effect=render_pages,
            ) as render,
            patch(
                "app.services.media_content_service.analyze_image",
                side_effect=analyze,
            ) as analyze_mock,
            patch("app.services.media_content_service.settings") as mock_settings,
        ):
            mock_settings.pdf_ocr_batch_pages = 8
            mock_settings.pdf_ocr_dpi = 160
            mock_settings.document_extraction_timeout_seconds = 180
            result = await extract_scanned_pdf_text(source)

        assert render.await_count == 2
        assert analyze_mock.await_count == 10
        assert "## Page 1" in result
        assert "## Page 10" in result


class TestVideoTimeline:
    async def test_scene_change_extraction_failure_is_not_silently_ignored(
        self, tmp_path
    ):
        from app.services.media_content_service import (
            MediaProcessingError,
            extract_video_frames,
        )

        source = tmp_path / "video.mp4"
        source.write_bytes(b"video")

        async def run_process(*args, **_kwargs):
            output = str(args[-1])
            if "interval-" in output:
                interval = tmp_path / "visual-timeline" / "interval-000001.jpg"
                interval.parent.mkdir(exist_ok=True)
                interval.write_bytes(b"jpg")
                return 0, ""
            return (
                1,
                "ffmpeg build banner\n"
                + "configuration " * 50
                + "\nscene decoder failed",
            )

        with (
            patch(
                "app.services.media_content_service.shutil.which", return_value="ffmpeg"
            ),
            patch(
                "app.services.media_content_service._run_process",
                new_callable=AsyncMock,
                side_effect=run_process,
            ),
            patch("app.services.media_content_service.settings") as mock_settings,
        ):
            mock_settings.video_frame_interval_seconds = 30
            mock_settings.video_scene_threshold = 0.3
            with pytest.raises(MediaProcessingError, match="scene decoder failed"):
                await extract_video_frames(source)

    async def test_clip_shorter_than_the_interval_falls_back_to_one_frame(
        self, tmp_path
    ):
        """fps=1/30 yields nothing for a 10 second video note, and ffmpeg still exits 0."""
        from app.services.media_content_service import extract_video_frames

        source = tmp_path / "round.mp4"
        source.write_bytes(b"video")
        timeline = tmp_path / "visual-timeline"
        calls: list[tuple[str, ...]] = []

        async def run_process(*args, **_kwargs):
            calls.append(args)
            output = str(args[-1])
            if "interval-%06d" in output:
                return 0, ""
            if "interval-000001" in output:
                timeline.mkdir(exist_ok=True)
                (timeline / "interval-000001.jpg").write_bytes(b"jpg")
                return 0, ""
            return 0, ""

        with (
            patch(
                "app.services.media_content_service.shutil.which", return_value="ffmpeg"
            ),
            patch(
                "app.services.media_content_service._run_process",
                new_callable=AsyncMock,
                side_effect=run_process,
            ),
            patch("app.services.media_content_service.settings") as mock_settings,
        ):
            mock_settings.video_frame_interval_seconds = 30
            mock_settings.video_scene_threshold = 0.3
            frames = await extract_video_frames(source)

        assert frames, "a short clip must still yield a frame"
        assert any("-frames:v" in call for call in calls), (
            "expected the single-frame fallback"
        )
        # Modern ffmpeg rejects limited-range YUV for MJPEG. Both sampling paths
        # and the short-clip fallback must request a valid JPEG pixel format.
        assert all("-pix_fmt" in call and "yuvj420p" in call for call in calls)
