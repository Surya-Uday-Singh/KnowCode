# KnowCode

Context-aware Python code summarization tailored by team role (Developer, QA Engineer, Product Manager).
COSC 4426 – LLMs in Software Development.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add your DeepSeek API key
```

## Development

```bash
pytest          # run the tests (no API key or network needed)
ruff check .    # lint
ruff format .   # format
```
