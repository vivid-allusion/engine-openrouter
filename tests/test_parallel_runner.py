"""N-in-flight proofs — mirrored from the gold engine's stub suite (phase_3)."""

import sys
import threading
import time
from datetime import datetime as _dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_openrouter import Engine, EngineError, InputFile  # noqa: E402


class _Frozen(_dt):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 18, 12, 0, 0)


def inputs_stub(n=3):
    return [InputFile(path=Path(f"b{i}.md"), prompt=f"p{i}") for i in range(n)]


def base_profile(**extra):
    return {"endpoint": "test/model", "media_type": "image", **extra}


def run_engine(profile, out_dir, impl, progress=None, n=3, frozen=True):
    mock_openai = MagicMock()
    mock_openai.openai = mock_openai
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = impl
    mock_openai.OpenAI.return_value = mock_client
    recorder = progress if progress is not None else []
    patches = [
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test0000000000000000"}),
        patch.dict("sys.modules", {"openai": mock_openai}),
    ]
    if frozen:
        patches.append(patch("engine_openrouter.engine.datetime", _Frozen))
    with patches[0], patches[1]:
        ctx = patches[2] if frozen else _NullCtx()
        with ctx:
            engine = Engine(profile, out_dir, on_progress=recorder.append)
            return engine.run(inputs_stub(n)), recorder, mock_client


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def ok_impl(model=None, messages=None, **kwargs):
    time.sleep(0.02 + (hash(threading.current_thread()) % 11) * 0.002)
    kind = MagicMock()
    kind.choices = [MagicMock(message=MagicMock(content="answer"))]
    return kind


class TestParallelRunner:
    def test_result_set_and_order_identical_to_own_concurrency_one(self, tmp_path):
        seq_results, _, _ = run_engine(base_profile(), tmp_path / "seq", ok_impl)
        par_results, _, _ = run_engine(
            base_profile(concurrency=3), tmp_path / "par", ok_impl
        )
        assert [r.status for r in par_results] == [r.status for r in seq_results]
        assert [r.source_path for r in par_results] == [
            r.source_path for r in seq_results
        ]
        assert [r.path.name for r in par_results] == [r.path.name for r in seq_results]
        assert len(par_results) == 3

    def test_more_than_one_job_in_flight(self, tmp_path):
        state = {"cur": 0, "max": 0, "lock": threading.Lock()}

        def watching(model=None, messages=None, **kwargs):
            with state["lock"]:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            try:
                time.sleep(0.04)
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(content="a"))]
                )
            finally:
                with state["lock"]:
                    state["cur"] -= 1

        run_engine(base_profile(concurrency=3), tmp_path, watching)
        assert state["max"] >= 2

    def test_never_more_than_n_in_flight(self, tmp_path):
        state = {"cur": 0, "max": 0, "lock": threading.Lock()}

        def watching(model=None, messages=None, **kwargs):
            with state["lock"]:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            try:
                time.sleep(0.04)
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(content="a"))]
                )
            finally:
                with state["lock"]:
                    state["cur"] -= 1

        run_engine(base_profile(concurrency=2), tmp_path, watching)
        assert state["max"] <= 2

    def test_failing_item_leaves_in_flight_siblings_intact(self, tmp_path):
        def impl(model=None, messages=None, **kwargs):
            time.sleep(0.02)
            text = str(messages)
            if "boom" in text:
                raise RuntimeError("openrouter blew up")
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content="a"))]
            )

        files = [
            InputFile(path=Path("b0.md"), prompt="fine"),
            InputFile(path=Path("b1.md"), prompt="boom"),
            InputFile(path=Path("b2.md"), prompt="fine"),
        ]
        progress = []
        mock_openai = MagicMock()
        mock_openai.openai = mock_openai
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = impl
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict(
            "os.environ", {"OPENROUTER_API_KEY": "sk-or-test0000000000000000"}
        ), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ), patch("engine_openrouter.engine.datetime", _Frozen):
            engine = Engine(base_profile(concurrency=3), tmp_path, on_progress=progress.append)
            results = engine.run(files)
        statuses = [r.status for r in results]
        assert "error" in statuses and statuses.count("ok") == 2

    def test_concurrency_never_in_call_kwargs(self, tmp_path):
        results, _, mock_client = run_engine(
            base_profile(concurrency=3), tmp_path, ok_impl
        )
        for call in mock_client.chat.completions.create.call_args_list:
            assert "concurrency" not in str(call.kwargs)
        assert len(results) == 3

    def test_invalid_concurrency_raises_loudly(self, tmp_path):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test0000000000000000"}):
            engine = Engine(base_profile(concurrency=0), tmp_path)
            with pytest.raises(EngineError, match="concurrency"):
                engine.run([])
