import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_openrouter import Engine, EngineError, InputFile, OutputFile, ProgressEvent  # noqa: E402
import engine_openrouter.metadata as meta  # noqa: E402


class TestDatatypes:
    def test_inputfile_defaults(self):
        f = InputFile(path=Path("test.md"), prompt="hello")
        assert f.path == Path("test.md")
        assert f.prompt == "hello"
        assert f.reference_urls == []
        assert f.metadata == {}

    def test_outputfile_defaults(self):
        o = OutputFile(bullet_path=Path("test.md"))
        assert o.bullet_path == Path("test.md")
        assert o.path is None
        assert o.status == "ok"
        assert o.error_msg == ""
        assert o.media_type == ""
        assert o.metadata == {}

    def test_outputfile_error_status(self):
        o = OutputFile(
            bullet_path=Path("test.md"),
            status="error",
            error_msg="timeout",
            media_type="text",
        )
        assert o.status == "error"
        assert o.error_msg == "timeout"
        assert o.media_type == "text"

    def test_progress_event_defaults(self):
        e = ProgressEvent(message="processing")
        assert e.message == "processing"
        assert e.level == "info"

    def test_engine_error_is_exception(self):
        with pytest.raises(EngineError):
            raise EngineError("test")


class TestMetadata:
    def test_provider_name(self):
        assert meta.PROVIDER_NAME == "OpenRouter"

    def test_platform(self):
        assert meta.PLATFORM == "openrouter"

    def test_api_key_env_var(self):
        assert meta.API_KEY_ENV_VAR == "OPENROUTER_API_KEY"

    def test_api_key_pattern(self):
        assert meta.API_KEY_PATTERN == r"^sk-or-[A-Za-z0-9-]{20,}$"

    def test_provider_homepage(self):
        assert meta.PROVIDER_HOMEPAGE == "https://openrouter.ai"

    def test_metadata_matches_engine_class(self):
        assert Engine.PLATFORM == meta.PLATFORM
        assert Engine.PROVIDER_NAME == meta.PROVIDER_NAME
        assert Engine.API_KEY_ENV_VAR == meta.API_KEY_ENV_VAR
        assert Engine.API_KEY_PATTERN == meta.API_KEY_PATTERN


class TestEngineInitPurity:
    def test_init_stores_attributes(self):
        profile = {"endpoint": "test/model", "media_type": "text"}
        engine = Engine(profile, "/tmp/out")
        assert engine._profile == profile
        assert engine._output_dir == Path("/tmp/out")
        assert engine._api_key is None
        assert engine._on_progress is None
        assert engine._prefix == ""
        assert engine._suffix == ""

    def test_init_extracts_prefix_suffix(self):
        profile = {
            "endpoint": "test/model",
            "prompt_prefix": "Write a story about ",
            "prompt_suffix": " in 100 words.",
        }
        engine = Engine(profile, "/tmp/out")
        assert engine._prefix == "Write a story about "
        assert engine._suffix == " in 100 words."


class TestEnginePreflight:
    def test_missing_endpoint_raises(self):
        engine = Engine({"media_type": "text"}, "/tmp/out")
        with pytest.raises(EngineError, match="Missing 'endpoint'"):
            engine.run([])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            engine = Engine({"endpoint": "test/model"}, "/tmp/out")
            with pytest.raises(EngineError, match="not set"):
                engine.run([])

    def test_openai_not_installed_raises(self):
        with patch.dict("sys.modules", {"openai": None}):
            engine = Engine({"endpoint": "test/model"}, "/tmp/out")
            with pytest.raises(EngineError, match="openai SDK not installed"):
                engine.run([])


class TestEngineRun:
    def test_empty_inputs(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {"endpoint": "test/model", "media_type": "text"},
                    tmp_path,
                )
                results = engine.run([])
                assert results == []

    def test_run_calls_progress_callback(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Generated text"
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            mock_openai.OpenAI.return_value = mock_client

            progress_calls = []
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {"endpoint": "test/model", "media_type": "text"},
                    tmp_path,
                    on_progress=progress_calls.append,
                )
                engine.run([InputFile(path=Path("b.md"), prompt="test")])

            assert len(progress_calls) >= 1
            assert "Processing bullet" in progress_calls[0]

    def test_applies_prefix_suffix(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Result"
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {
                        "endpoint": "test/model",
                        "media_type": "text",
                        "prompt_prefix": "PREFIX: ",
                        "prompt_suffix": " :SUFFIX",
                    },
                    tmp_path,
                )
                engine.run([InputFile(path=Path("b.md"), prompt="hello")])
                call_args = mock_client.chat.completions.create.call_args
                messages = call_args[1]["messages"]
                assert "PREFIX: hello :SUFFIX" == messages[0]["content"]

    def test_empty_prompt_after_prefix_suffix_is_error(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {
                        "endpoint": "test/model",
                        "media_type": "text",
                        "prompt_prefix": "",
                        "prompt_suffix": "",
                    },
                    tmp_path,
                )
                results = engine.run([InputFile(path=Path("b.md"), prompt="  ")])
                assert len(results) == 1
                assert results[0].status == "error"
                assert "Empty prompt" in results[0].error_msg

    def test_per_bullet_error_returns_error_outputfile(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError(
                "API timeout"
            )
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {"endpoint": "test/model", "media_type": "text"},
                    tmp_path,
                )
                results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
                assert len(results) == 1
                assert results[0].status == "error"
                assert "API timeout" in results[0].error_msg

    def test_partial_success_mixed_batch(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()

            def side_effect(**kwargs):
                idx = kwargs.get("_idx", 0)
                if idx == 1:
                    raise RuntimeError("fail")
                mock_choice = MagicMock()
                mock_choice.message.content = "ok"
                return MagicMock(choices=[mock_choice])

            mock_client.chat.completions.create.side_effect = side_effect
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {"endpoint": "test/model", "media_type": "text"},
                    tmp_path,
                )
                bullets = [
                    InputFile(path=Path(f"b{i}.md"), prompt="test")
                    for i in range(3)
                ]
                results = engine.run(bullets)
                statuses = [r.status for r in results]
                assert statuses.count("ok") == 2
                assert statuses.count("error") == 1

    def test_text_output_saved_as_txt(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Generated text output"
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {"endpoint": "test/model", "media_type": "text"},
                    tmp_path,
                )
                results = engine.run(
                    [InputFile(path=Path("b.md"), prompt="test")]
                )
                assert len(results) == 1
                assert results[0].status == "ok"
                assert results[0].path is not None
                assert results[0].path.suffix == ".txt"
                assert results[0].path.read_text() == "Generated text output"

    def test_none_content_returns_error(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = None
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {"endpoint": "test/model", "media_type": "text"},
                    tmp_path,
                )
                results = engine.run(
                    [InputFile(path=Path("b.md"), prompt="test")]
                )
                assert len(results) == 1
                assert results[0].status == "error"
                assert "No content" in results[0].error_msg

    def test_vision_media_type_includes_image_urls(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Vision result"
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {"endpoint": "test/vision-model", "media_type": "vision"},
                    tmp_path,
                )
                engine.run([
                    InputFile(
                        path=Path("b.md"),
                        prompt="Describe this",
                        reference_urls=["https://img.example.com/photo.jpg"],
                    )
                ])
                messages = mock_client.chat.completions.create.call_args[1]["messages"]
                user_msg = messages[-1]
                assert isinstance(user_msg["content"], list)
                assert user_msg["content"][0]["type"] == "text"
                assert user_msg["content"][1]["type"] == "image_url"
                assert user_msg["content"][1]["image_url"]["url"] == "https://img.example.com/photo.jpg"

    def test_system_prompt_added_when_present(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            mock_openai = MagicMock()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Result"
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            mock_openai.OpenAI.return_value = mock_client
            with patch.dict("sys.modules", {"openai": mock_openai}):
                engine = Engine(
                    {
                        "endpoint": "test/model",
                        "media_type": "text",
                        "system_prompt": "You are helpful.",
                    },
                    tmp_path,
                )
                engine.run([InputFile(path=Path("b.md"), prompt="test")])
                messages = mock_client.chat.completions.create.call_args[1]["messages"]
                assert messages[0]["role"] == "system"
                assert messages[0]["content"] == "You are helpful."


class TestImports:
    def test_init_exports_all_names(self):
        from engine_openrouter import (
            Engine,
            EngineError,
            InputFile,
            OutputFile,
            ProgressEvent,
        )
        assert Engine is not None
        assert InputFile is not None
        assert OutputFile is not None
        assert ProgressEvent is not None
        assert EngineError is not None
