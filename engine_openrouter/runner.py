"""N-in-flight execution — mirrors the gold engine's runner.

`run_batch` caps jobs in flight structurally: at most the profile's
`concurrency` futures are submitted at any moment. The per-item body is
this engine's sequential body, verbatim (build messages, create the
completion, save the text output); results are written by input index;
one failing item never disturbs jobs still in flight.
"""

import concurrent.futures
import threading

from .datatypes import OutputFile


class BatchState:
    def __init__(self):
        self._stop = False
        self._lock = threading.Lock()

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stop

    def halt(self):
        with self._lock:
            self._stop = True

    def note_success(self):
        pass


def run_item(engine, client, item, idx, total, state) -> list[OutputFile]:
    media_type = engine._profile.get("media_type", "")
    engine._emit(f"Processing Markdown file {idx + 1}/{total}...")
    prompt = f"{engine._prefix}{item.prompt}{engine._suffix}".strip()
    if not prompt:
        return [
            OutputFile(
                source_path=item.path,
                status="error",
                error_msg="Empty prompt after applying prefix/suffix",
                media_type=media_type,
            )
        ]
    try:
        messages = engine._build_messages(prompt, item, media_type)
        params = dict(engine._profile.get("parameters", {}))
        completion = client.chat.completions.create(
            model=engine._profile["endpoint"],
            messages=messages,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 1024),
        )

        content = completion.choices[0].message.content
        if not content:
            return [
                OutputFile(
                    source_path=item.path,
                    status="error",
                    error_msg="No content returned from OpenRouter",
                    media_type=media_type,
                )
            ]
        state.note_success()
        ts = engine._now_string()
        dest = engine._output_dir / f"{ts}-{item.path.stem}-{idx}.txt"
        dest.write_text(content, encoding="utf-8")
        return [
            OutputFile(
                source_path=item.path,
                path=dest,
                status="ok",
                media_type=media_type,
            )
        ]
    except Exception as exc:
        return [
            OutputFile(
                source_path=item.path,
                status="error",
                error_msg=str(exc),
                media_type=media_type,
            )
        ]


def run_batch(engine, inputs) -> list[OutputFile]:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=engine._resolve_api_key(),
    )

    state = BatchState()
    results: list[list[OutputFile] | None] = [None] * len(inputs)
    inflight_cap = max(1, engine._concurrency)

    with concurrent.futures.ThreadPoolExecutor(max_workers=inflight_cap) as pool:
        pending: dict[concurrent.futures.Future, int] = {}
        next_idx = 0
        while pending or (next_idx < len(inputs) and not state.is_stopped()):
            while (
                not state.is_stopped()
                and next_idx < len(inputs)
                and len(pending) < inflight_cap
            ):
                future = pool.submit(
                    run_item, engine, client, inputs[next_idx], next_idx, len(inputs), state
                )
                pending[future] = next_idx
                next_idx += 1
            if not pending:
                break
            done, _ = concurrent.futures.wait(pending, return_when="FIRST_COMPLETED")
            for future in done:
                results[pending.pop(future)] = future.result()

    return [out for item in results if item is not None for out in item]
