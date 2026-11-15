# Asama.ai AI-engineering assignment

Two independent, fully working Python projects:

| Folder | Problem | Run | Test |
|---|---|---|---|
| `agent_network/` | 1. A social network for AI agents: relevance routing, webhooks, comments, feedback | `pip install -r requirements-dev.txt && python scripts/demo.py` | `python -m pytest -q` (38 tests) |
| `hw_assistant/` | 2. A Linux hardware Q&A assistant: safe tool calling, local LLM or offline engine | `python3 -m hwqa` | `python3 -m pytest -q` (35 tests) |

Both work offline with no API keys. Each one can use a real LLM (a local Ollama model, or OpenAI) when one is configured.
See each folder's README for the design and the trade-offs.

**Deploying:** see [DEPLOYMENT.md](DEPLOYMENT.md).
- **Problem 1** ships as a Docker image, with docker compose, Render and Fly.io configs.
- **Problem 2** ships as a pip/pipx package, a single-file `hwqa.pyz` and an install script.
- CI/CD is in `.github/workflows/`.
