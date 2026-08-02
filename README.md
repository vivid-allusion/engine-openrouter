# engine-openrouter — OpenRouter Engine for studiolot

Engine wrapping the [OpenRouter](https://openrouter.ai) API via the
OpenAI-compatible SDK behind studiolot's uniform Engine interface.
Vehicles call this Engine instead of importing the OpenAI SDK directly.
OpenRouter routes to 300+ LLM and vision models (Claude, ChatGPT, Gemini,
DeepSeek, etc.) through a single API.

## Quick Start

```bash
# Clone into a studiolot project
git clone https://github.com/vivid-allusion/engine-openrouter.git \
  00_APPLICATIONS/ENGINES/engine-openrouter/

# Or pip install
pip install git+https://github.com/vivid-allusion/engine-openrouter.git
```

## Usage

```python
from engine_openrouter import Engine, InputFile
from pathlib import Path

profile = {
    "platform": "openrouter",
    "endpoint": "anthropic/claude-3.5-sonnet",
    "media_type": "text",
    "parameters": {"temperature": 0.7, "max_tokens": 1024},
}

engine = Engine(profile, "/tmp/output", api_key="sk-or-...")
results = engine.run([InputFile(path=Path("prompt.md"), prompt="Write a haiku")])

for r in results:
    print(r.status, r.path)
```

## API Key

Set `OPENROUTER_API_KEY` in your environment or `.env` file:

```bash
export OPENROUTER_API_KEY=sk-or-v1-your_key_here
```

Get a key at https://openrouter.ai/keys

## Repo Structure

```
engine-openrouter/
├── engine.py          ← Engine class wrapping openai SDK (OpenRouter base_url)
├── datatypes.py       ← InputFile, OutputFile, ProgressEvent, EngineError
├── metadata.py        ← Zero-dependency identity constants
├── __init__.py        ← Re-exports
├── pyproject.toml     ← pip install definition
├── requirements.txt   ← openai>=1.0
├── endpoints/
│   ├── IMG-Models/    ← .gitkeep (OpenRouter is text/vision only)
│   ├── VID-Models/    ← .gitkeep (OpenRouter is text/vision only)
│   ├── TXT-Models/    ← 10 LLM models (claude, deepseek, gemini, gpt, kimi)
│   └── Vision-Models/ ← 3 vision models (kimi)
└── tests/
    └── test_engine.py ← 23 unit tests
```

## Supported Models

### TXT-Models (LLM)
- anthropic/claude-3.5-sonnet, deepseek/deepseek-v3, google/gemini-2.0-flash,
  google/gemini-2.5-flash, google/gemini-3.1-flash, google/gemini-3.1-pro,
  openai/gpt-4o, moonshotai/kimi-k2.5, moonshotai/kimi-k2.6,
  moonshotai/kimi-k2-thinking

### Vision-Models
- moonshotai/kimi-k2.5, moonshotai/kimi-k2.6, moonshotai/kimi-k2-thinking

More models can be added by dropping TOML files into `endpoints/TXT-Models/`
or `endpoints/Vision-Models/`.

## Contract

See [ENGINE_CONTRACT.md](https://github.com/vivid-allusion/studiolot/blob/main/docs/architecture/ENGINE_CONTRACT.md)
in the studiolot repo for the full Engine specification.
