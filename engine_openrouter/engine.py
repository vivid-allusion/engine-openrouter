import json
import os
from datetime import datetime
from pathlib import Path

from .datatypes import EngineError, InputFile, OutputFile, ProgressEvent


class Engine:
    PLATFORM: str = "openrouter"
    PROVIDER_NAME: str = "OpenRouter"
    PROVIDER_HOMEPAGE: str = "https://openrouter.ai"
    API_KEY_ENV_VAR: str = "OPENROUTER_API_KEY"
    API_KEY_PATTERN: str = r"^sk-or-[A-Za-z0-9-]{20,}$"

    def __init__(
        self,
        profile: dict,
        output_dir: str | Path,
        api_key: str | None = None,
        on_progress: "Callable[[str], None] | None" = None,
    ):
        self._profile = profile
        self._output_dir = Path(output_dir)
        self._api_key = api_key
        self._on_progress = on_progress
        self._prefix = profile.get("prompt_prefix", "")
        self._suffix = profile.get("prompt_suffix", "")

    def run(self, inputs: list[InputFile]) -> list[OutputFile]:
        self._validate_preflight()

        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._resolve_api_key(),
        )
        endpoint = self._profile["endpoint"]
        params = dict(self._profile.get("parameters", {}))
        media_type = self._profile.get("media_type", "")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        results: list[OutputFile] = []
        total = len(inputs)

        for idx, item in enumerate(inputs):
            self._emit(f"Processing Markdown file {idx + 1}/{total}...")
            prompt = f"{self._prefix}{item.prompt}{self._suffix}".strip()
            if not prompt:
                output = OutputFile(
                    source_path=item.path,
                    status="error",
                    error_msg="Empty prompt after applying prefix/suffix",
                    media_type=media_type,
                )
                results.append(output)
                continue

            try:
                messages = self._build_messages(prompt, item, media_type)
                completion = client.chat.completions.create(
                    model=endpoint,
                    messages=messages,
                    temperature=params.get("temperature", 0.7),
                    max_tokens=params.get("max_tokens", 1024),
                )

                content = completion.choices[0].message.content
                if not content:
                    output = OutputFile(
                        source_path=item.path,
                        status="error",
                        error_msg="No content returned from OpenRouter",
                        media_type=media_type,
                    )
                else:
                    ts = datetime.now().strftime("%y%m%d_%H%M%S")
                    dest = self._output_dir / f"{ts}-{item.path.stem}-{idx}.txt"
                    dest.write_text(content, encoding="utf-8")
                    output = OutputFile(
                        source_path=item.path,
                        path=dest,
                        status="ok",
                        media_type=media_type,
                    )

            except Exception as exc:
                output = OutputFile(
                    source_path=item.path,
                    status="error",
                    error_msg=str(exc),
                    media_type=media_type,
                )

            results.append(output)

        return results

    def _validate_preflight(self):
        if not self._profile.get("endpoint"):
            raise EngineError("Missing 'endpoint' in profile")
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError:
            raise EngineError(
                "openai SDK not installed. Run: pip install openai"
            ) from None
        if not self._resolve_api_key():
            raise EngineError(
                f"{self.API_KEY_ENV_VAR} not set in environment or .env file"
            )

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return os.environ.get(self.API_KEY_ENV_VAR, "")

    def _build_messages(
        self, prompt: str, item: InputFile, media_type: str
    ) -> list[dict]:
        system = self._profile.get("system_prompt", "")
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})

        if media_type == "vision" and item.reference_urls:
            content_blocks: list[dict] = [
                {"type": "text", "text": prompt}
            ]
            for url in item.reference_urls:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
            messages.append({"role": "user", "content": content_blocks})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    def _emit(self, message: str, level: str = "info"):
        if self._on_progress:
            self._on_progress(message)
