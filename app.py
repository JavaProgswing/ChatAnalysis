"""
app.py -- Flask server for the Chat Analysis Dashboard.
Run: python app.py
Then open: http://localhost:5050
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

load_dotenv()

from analyzer.data_loader import load_all_data
from analyzer.stats       import compute_stats, get_chat_summaries
from analyzer import charts as chart_module
from analyzer import ai_profiler

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

CACHE_PATH = Path(__file__).parent / "cache"
CACHE_PATH.mkdir(exist_ok=True)

# ── In-memory cache (loaded once on first request) ───────────────────────────
_cache: dict = {}


def _ensure_loaded():
    if _cache:
        return
    print("[*] Loading all chat data...")
    df, meta       = load_all_data()
    stats          = compute_stats(df, meta)
    chat_sums      = get_chat_summaries(df)
    all_charts     = chart_module.build_all(stats)
    _cache["df"]       = df
    _cache["meta"]     = meta
    _cache["stats"]    = stats
    _cache["chat_summaries"] = chat_sums
    _cache["charts"]   = all_charts
    if len(df) == 0:
        print("\n[WARNING] Whoops! No chat data found.")
        print("[WARNING] Please check your .env file and ensure DATA_EXPORT_PATH is correct.\n")
    else:
        print(f"[OK] Loaded {len(df):,} messages.")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/stats")
def api_stats():
    _ensure_loaded()
    s = _cache["stats"]
    # Return everything except the large all_text_sample
    safe = {k: v for k, v in s.items() if k != "all_text_sample"}
    return jsonify(safe)


@app.route("/api/charts")
def api_charts():
    _ensure_loaded()
    return jsonify(_cache["charts"])


@app.route("/api/ai/profile")
def api_ai_profile():
    """Run the AI analysis. Cached to disk so it's only run once per cache clear."""
    import hashlib
    profile_cache = CACHE_PATH / "ai_profile.json"
    
    _ensure_loaded()
    s = _cache["stats"]
    data_sig = f"{s.get('total_messages', 0)}_{s.get('date_range', {}).get('start')}_{s.get('date_range', {}).get('end')}"
    current_hash = hashlib.md5(data_sig.encode()).hexdigest()

    force = request.args.get("force", "true").lower() == "true"
    if profile_cache.exists() and not force:
        try:
            with open(profile_cache, encoding="utf-8") as f:
                cached_data = json.load(f)
                if cached_data.get("_data_hash") == current_hash:
                    return jsonify(cached_data)
        except Exception:
            pass

    model      = request.args.get("model", "gemma3:4b")
    max_retries = int(request.args.get("retries", "2"))
    result = ai_profiler.analyze(
        _cache["df"], _cache["stats"], _cache["meta"],
        model=model, max_retries=max_retries,
    )

    if result.get("success"):
        result["_data_hash"] = current_hash
        with open(profile_cache, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return jsonify(result)


@app.route("/api/ai/models")
def api_ai_models():
    """List available AI backends."""
    import os
    models = {
        "ollama": ["gemma3:4b", "llama3", "mistral"],
        "cloud": {
            "gemini": bool(os.environ.get("GEMINI_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
        }
    }
    # Try to get actual Ollama model list
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.ok:
            models["ollama"] = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return jsonify(models)


@app.route("/api/ai/chat", methods=["POST"])
def api_ai_chat():
    """Interactive chat with your data."""
    _ensure_loaded()
    
    data = request.json or {}
    history = data.get("history", [])
    model = data.get("model", "gemma3:4b")
    context_summaries = data.get("context_summaries", {})
    
    if not history:
        return jsonify({"success": False, "error": "No chat history provided."})
        
    result = ai_profiler.chat_with_data(
        history, _cache["df"], _cache["stats"], _cache["meta"], model=model, context_summaries=context_summaries
    )
    
    return jsonify(result)


@app.route("/api/ai/roast")
def api_ai_roast():
    """Generate a comedic roast of the user's chat habits."""
    _ensure_loaded()
    model = request.args.get("model", "gemma3:4b")
    result = ai_profiler.roast_user(
        _cache["df"], _cache["stats"], _cache["meta"], model=model
    )
    return jsonify(result)


@app.route("/api/chats")
def api_chats():
    """Returns the meta-summaries of all chats."""
    _ensure_loaded()
    return jsonify(_cache.get("chat_summaries", {}))


@app.route("/api/ai/summarize_chat")
def api_ai_summarize_chat():
    """Generates an AI narrative summary for a specific channel."""
    _ensure_loaded()
    channel = request.args.get("channel")
    model = request.args.get("model", "gemma3:4b")
    
    if not channel:
        return jsonify({"success": False, "error": "No channel provided"})
        
    result = ai_profiler.summarize_single_chat(channel, _cache["df"], model=model)
    return jsonify(result)



@app.route("/api/ai/reset")
def api_ai_reset():
    """Delete cached AI profile so it is regenerated on next call."""
    p = CACHE_PATH / "ai_profile.json"
    if p.exists():
        p.unlink()
    return jsonify({"ok": True})

@app.route("/api/download/filtered")
def download_filtered():
    """Download the completely cleaned, deduped dataset."""
    filtered_path = Path(__file__).parent / "filtered_data" / "all_clean_messages.csv"
    if filtered_path.exists():
        return send_from_directory(filtered_path.parent, filtered_path.name, as_attachment=True)
    return jsonify({"error": "Filtered data not generated yet."}), 404

if __name__ == "__main__":
    print("[*] Starting Chat Analysis Dashboard on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
