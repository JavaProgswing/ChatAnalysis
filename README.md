# YourMind — Chat Analysis Dashboard

A concise tool to analyze chat exports from WhatsApp, Discord, ChatGPT, and Claude. Produces usage statistics, visualizations, and an optional AI personality profile using local or cloud LLMs.

Key points

- Purpose: load and normalize chat exports, compute metrics, create Plotly visualizations, and generate an AI profile when requested.
- Data sources: WhatsApp (text), Discord (JSON), GPT/Claude (JSON).
- AI support: local models via Ollama or cloud models (Gemini/OpenAI). Results are cached in `cache/ai_profile.json`.

Minimal project layout

- `app.py` — Flask backend server
- `analyzer/` — data parsing, stats, charts, and AI profiling modules
- `static/index.html` — frontend dashboard
- `cache/` — cached AI profile and artifacts

Prerequisites and install

- Python 3.9 or newer
- Optional: Ollama for local models
- Install runtime dependencies:

```bash
pip install flask pandas numpy plotly wordcloud python-dotenv requests
```

Configuration and data

- Configure `DATA_EXPORT_PATH` in `.env` to point to your exports (default previously used: `D:/DataExport`).
- Place exports under subfolders: `WHATSAPP/`, `DISCORD/`, `GPT/`, `CLAUDE/` as appropriate.

Run

- Start the server:

```bash
python app.py
```

- Open http://localhost:5050

Notes

- The AI profiler is optional. For local models run `ollama serve` and select a supported model in the UI. Cloud models require API keys in `.env`.
