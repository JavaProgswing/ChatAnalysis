"""
ai_profiler.py — build a rich personality-analysis prompt and call Ollama or a cloud LLM.

Identity facts are INFERRED from the chat data itself — nothing is hardcoded.
Supports: Ollama (local), Gemini (Google), OpenAI.
"""

import os
import re
import json
import time
import random
import requests
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:4b"

URL_RE = re.compile(r"http\S+|www\S+")


# ── Identity inference ────────────────────────────────────────────────────────
def _infer_identity(df, metadata: dict) -> dict:
    """
    Derive who-you-are facts purely from the data itself.
    Returns a dict of inferred facts to embed in the prompt.
    """
    if len(df) == 0:
        return {
            "first_msg": "N/A",
            "last_msg": "N/A",
            "days_span": 0,
            "gpt_convos": 0,
            "claude_convos": 0,
            "wa_contacts": [],
            "dc_channels_sample": [],
            "gpt_topics_sample": [],
            "claude_topics_sample": []
        }

    facts: dict = {}

    # WA display name (most common 'me' author in WA)
    wa_me = df[(df["source"] == "WhatsApp") & (df["is_me"] == True)]["author"]
    if len(wa_me):
        facts["wa_name"] = wa_me.mode().iloc[0]

    # Discord username
    dc_me = df[(df["source"] == "Discord") & (df["is_me"] == True)]["author"]
    if len(dc_me):
        facts["dc_username"] = dc_me.mode().iloc[0]

    # GPT + Claude volume
    facts["gpt_convos"]    = df[df["source"] == "ChatGPT"]["channel"].nunique()
    facts["claude_convos"] = df[df["source"] == "Claude"]["channel"].nunique()

    # Date range
    facts["first_msg"] = df["timestamp"].min().strftime("%b %Y")
    facts["last_msg"]  = df["timestamp"].max().strftime("%b %Y")
    facts["days_span"] = (df["timestamp"].max() - df["timestamp"].min()).days

    # Top chat contacts (WA, excluding numbers/unknown)
    wa_contacts = (
        df[(df["source"] == "WhatsApp") & (~df["is_me"])]
        ["author"].value_counts().head(20).index.tolist()
    )
    # Prefer names (contain letters, not just digits)
    name_contacts = [c for c in wa_contacts if re.search(r"[A-Za-z]", c)]
    facts["wa_contacts"] = name_contacts[:10]

    # Discord server names (from channel field: "ServerName - #channel")
    dc_channels = df[df["source"] == "Discord"]["channel"].unique().tolist()
    facts["dc_channels_sample"] = dc_channels[:8]

    # GPT topics sample
    facts["gpt_topics_sample"] = metadata.get("gpt_titles", [])[:40]
    facts["claude_topics_sample"] = metadata.get("claude_topics", [])[:20]

    return facts


def _build_known_facts_block(facts: dict) -> str:
    """Build the KNOWN FACTS section from inferred data."""
    lines = ["KNOWN FACTS INFERRED FROM THE DATA (treat as ground truth):"]

    if "wa_name" in facts:
        lines.append(f"- WhatsApp display name: {facts['wa_name']}")
    if "dc_username" in facts:
        lines.append(f"- Discord username: {facts['dc_username']}")

    lines.append(f"- Data spans: {facts['first_msg']} → {facts['last_msg']} ({facts['days_span']} days)")
    lines.append(f"- ChatGPT conversations: {facts['gpt_convos']}")
    lines.append(f"- Claude conversations: {facts['claude_convos']}")

    if facts.get("wa_contacts"):
        lines.append(f"- Top WhatsApp contacts: {', '.join(facts['wa_contacts'][:8])}")

    if facts.get("dc_channels_sample"):
        lines.append(f"- Discord channels/servers used: {', '.join(facts['dc_channels_sample'][:5])}")

    if facts.get("gpt_topics_sample"):
        sample = facts["gpt_topics_sample"][:12]
        lines.append(f"- Sample of ChatGPT conversation topics: {'; '.join(sample)}")

    if facts.get("claude_topics_sample"):
        sample = facts["claude_topics_sample"][:8]
        lines.append(f"- Sample of Claude conversation topics: {'; '.join(sample)}")

    return "\n".join(lines)


# ── Message sampling ─────────────────────────────────────────────────────────
def _sample_messages(df, n_wa=80, n_dc=80, n_gpt=50, n_claude=50) -> list[str]:
    me_df = df[df["is_me"] == True]
    samples: list[str] = []

    def grab(source, n):
        msgs = me_df[me_df["source"] == source]["message"].tolist()
        random.shuffle(msgs)
        return msgs[:n]

    samples += grab("WhatsApp", n_wa)
    samples += grab("Discord",  n_dc)
    samples += grab("ChatGPT",  n_gpt)
    samples += grab("Claude",   n_claude)

    clean: list[str] = []
    for m in samples:
        m2 = URL_RE.sub("", str(m)).strip()
        if 8 <= len(m2) <= 400:
            clean.append(m2)

    random.shuffle(clean)
    return clean[:200]


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(df, stats: dict, metadata: dict) -> str:
    facts     = _infer_identity(df, metadata)
    known_blk = _build_known_facts_block(facts)
    msgs      = _sample_messages(df)

    ph        = stats.get("peak_hour", 22)
    emojis    = " ".join(e["emoji"] for e in stats.get("top_emojis", [])[:12])
    words     = ", ".join(w["word"] for w in stats.get("top_words", [])[:30])
    gpt_t     = "\n".join(f"  • {t}" for t in stats.get("gpt_titles", [])[:60])
    cl_t      = "\n".join(f"  • {t}" for t in stats.get("claude_topics", [])[:30])
    msg_block = "\n".join(f'{i+1}. "{m}"' for i, m in enumerate(msgs))
    dr        = stats.get("date_range", {})

    reply_sec = stats.get("avg_reply_seconds")
    reply_str = f"{reply_sec:.0f}s median reply time" if reply_sec else "unknown"

    top_chats = "\n".join(
        f"  • {k}: {v} msgs ({stats.get('top_channels_days',{}).get(k,'?')} unique days)"
        for k, v in list(stats.get("top_channels", {}).items())[:10]
    )

    return f"""You are an elite digital-behaviour analyst, psychologist, and life coach specialising in reading people through their digital footprint. You have years of someone's private chats, AI assistant conversations, emoji habits, and activity timestamps.

Your job: produce the MOST SPECIFIC, MOST INSIGHTFUL, MOST HONEST personality profile possible. Reference actual evidence from the data. Be bold. No generic platitudes. Write as if you genuinely know this person.

{known_blk}

=== QUANTITATIVE DATA ===
Messages sent (spam-cleaned): {stats.get('my_messages_clean', 0):,}
Total messages in dataset: {stats.get('total_messages', 0):,}
Spam messages removed: {stats.get('spam_removed', 0):,}
Platforms: {', '.join(stats.get('platform_breakdown', {}).keys())}
Period: {dr.get('start')} → {dr.get('end')} ({dr.get('days')} days)
Peak messaging hour: {ph:02d}:00
Night-owl (8pm–midnight): {stats.get('night_owl_pct', 0)}%
Late-night (midnight–4am): {stats.get('late_night_pct', 0)}% ({stats.get('late_night_messages', 0)} msgs)
Avg message length: {stats.get('avg_msg_length', 0)} chars / {stats.get('avg_word_count', 0)} words
Longest streak: {stats.get('max_streak_days', 0)} consecutive days
Unique people: {stats.get('unique_contacts', 0)}
Questions asked: {stats.get('questions_asked', 0)} ({stats.get('questions_pct', 0)}%)
Emojis sent: {stats.get('total_emojis_sent', 0)} ({stats.get('emoji_per_message', 0):.2f} per msg)
Most active day: {stats.get('most_active_day', '?')}, least: {stats.get('least_active_day', '?')}
Reply speed: {reply_str}

TOP CHATS (score = msgs × log(unique_days), spam-cleaned):
{top_chats}

TOP EMOJIS: {emojis}
TOP WORDS (stop-words + spam removed, social only): {words}

ChatGPT TOPICS:
{gpt_t}

Claude TOPICS:
{cl_t}

=== REAL MESSAGES (200 sampled, spam-cleaned) ===
{msg_block}

=== ANALYSIS TASK ===
Write an extremely detailed, specific, honest personality profile. Reference actual data evidence ("your {stats.get('max_streak_days',0)}-day streak shows...", "the late-night spikes at {ph}:00 combined with coding projects suggest..."). Use the known facts above — reference their WhatsApp name, Discord handle, actual contacts, GPT topics.

Return ONLY valid JSON. No markdown fences, no extra text. Schema:
{{
  "mbti": "XXXX",
  "mbti_reasoning": "2-3 specific sentences citing data evidence",
  "big_five": {{"openness":7,"conscientiousness":5,"extraversion":6,"agreeableness":7,"neuroticism":4}},
  "big_five_notes": {{"openness":"evidence sentence","conscientiousness":"sentence","extraversion":"sentence","agreeableness":"sentence","neuroticism":"sentence"}},
  "personality_traits": [
    "Trait 1 — evidence from data",
    "Trait 2 — evidence from data",
    "Trait 3 — evidence from data",
    "Trait 4 — evidence from data",
    "Trait 5 — evidence from data",
    "Trait 6 — evidence from data",
    "Trait 7 — evidence from data",
    "Trait 8 — evidence from data"
  ],
  "communication_style": "3 sentences about how they communicate: length, directness, humor, emoji use",
  "interests_and_passions": ["interest 1","interest 2","interest 3","interest 4","interest 5"],
  "social_patterns": "3 sentences about friendships and social circles",
  "digital_habits": "3 sentences about tech habits and AI usage patterns",
  "sleep_pattern": "1-2 sentences with specific hour evidence",
  "quirks": ["quirk 1 with evidence","quirk 2","quirk 3","quirk 4","quirk 5"],
  "character_summary": "5-6 sentence narrative portrait. Be vivid, specific, insightful. Reference their name, contacts, projects, habits.",
  "fun_facts": [
    "Fun fact 1 with specific number",
    "Fun fact 2 with specific number",
    "Fun fact 3 with specific number",
    "Fun fact 4 with specific number",
    "Fun fact 5 with specific number"
  ],
  "life_stage_guess": "Based on data: who is this person, what are they likely doing in life?",
  "energy_type": "Night owl or morning person? Introvert or extrovert in practice? Evidence-based.",
  "ai_relationship": "How do they use AI tools? What topics? What does this reveal about them?",
  "friend_dynamics": "Who are the closest contacts? What different social circles exist? How do they interact across them?",
  "college_vs_school": "If multiple social circles visible: how do different peer groups compare in messaging patterns?",
  "growth_arc": "Based on the {dr.get('days')} days of data: what arc or change is visible over time?"
}}"""


# ── JSON extraction ───────────────────────────────────────────────────────────
def _try_parse_json(raw: str) -> dict | None:
    """4-strategy JSON extraction from potentially messy model output."""
    # Strategy 1: direct
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Strategy 2: outermost { ... }
    try:
        start = raw.index('{')
        end   = raw.rindex('}') + 1
        return json.loads(raw[start:end])
    except Exception:
        pass

    # Strategy 3: fix trailing commas
    try:
        fixed = re.sub(r',\s*([}\]])', r'\1', raw)
        start = fixed.index('{')
        end   = fixed.rindex('}') + 1
        return json.loads(fixed[start:end])
    except Exception:
        pass

    # Strategy 4: field-by-field regex extraction
    profile: dict = {}
    str_fields = [
        "mbti", "mbti_reasoning", "communication_style", "social_patterns",
        "digital_habits", "sleep_pattern", "character_summary",
        "life_stage_guess", "energy_type", "ai_relationship",
        "friend_dynamics", "college_vs_school", "growth_arc",
    ]
    for field in str_fields:
        m = re.search(rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])', raw, re.DOTALL)
        if m:
            profile[field] = m.group(1).replace("\\n", " ").replace('\\"', '"')

    b5 = re.search(r'"big_five"\s*:\s*(\{[^}]+\})', raw)
    if b5:
        try:
            profile["big_five"] = json.loads(b5.group(1))
        except Exception:
            pass

    for field in ["personality_traits", "interests_and_passions", "quirks", "fun_facts"]:
        m = re.search(rf'"{field}"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if m:
            try:
                profile[field] = json.loads(m.group(1))
            except Exception:
                items = re.findall(r'"([^"]+)"', m.group(1))
                if items:
                    profile[field] = items

    return profile if len(profile) >= 3 else None


# ── Cloud model callers ───────────────────────────────────────────────────────
def _call_gemini(prompt: str) -> str:
    """Call Google Gemini via REST API. Set GEMINI_API_KEY env var."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 5000},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API. Set OPENAI_API_KEY env var."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 5000,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_ollama(prompt: str, model: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.65, "num_predict": 5000}},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


# ── Main analyze function ─────────────────────────────────────────────────────
def analyze(df, stats: dict, metadata: dict,
            model: str = DEFAULT_MODEL, max_retries: int = 2) -> dict:
    """
    Run the AI personality analysis.

    model options:
      - Any Ollama model name (e.g. "gemma3:4b", "llama3")
      - "gemini"  → Google Gemini Flash (needs GEMINI_API_KEY)
      - "openai"  → OpenAI GPT-4o-mini (needs OPENAI_API_KEY)
      - "openai:gpt-4o" etc.
    """
    prompt = build_prompt(df, stats, metadata)

    for attempt in range(1, max_retries + 1):
        raw = ""
        try:
            # ── Choose caller ─────────────────────────────────────────────────
            if model.startswith("gemini"):
                raw = _call_gemini(prompt)
            elif model.startswith("openai"):
                oai_model = model.split(":", 1)[1] if ":" in model else "gpt-4o-mini"
                raw = _call_openai(prompt, oai_model)
            else:
                raw = _call_ollama(prompt, model)

            # ── Clean output ──────────────────────────────────────────────────
            raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw.strip())
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                raw = json_match.group()

            profile = _try_parse_json(raw)
            if profile and len(profile) >= 3:
                return {"success": True, "profile": profile, "model": model, "attempts": attempt}

            print(f"[AI] Attempt {attempt}/{max_retries}: JSON extraction failed, retrying...")
            time.sleep(2)

        except Exception as e:
            print(f"[AI] Attempt {attempt}/{max_retries} error: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                return {"success": False, "error": str(e), "raw": raw[:2000], "attempts": attempt}

    return {
        "success": False,
        "error": f"Model returned unparseable output after {max_retries} attempts",
        "raw": raw[:3000],
        "attempts": max_retries,
    }


# ── Roast Mode ────────────────────────────────────────────────────────────────
def roast_user(df, stats: dict, metadata: dict, model: str = DEFAULT_MODEL) -> dict:
    """Generate a brutal AI roast based on user chat habits."""
    facts = _infer_identity(df, metadata)
    known_blk = _build_known_facts_block(facts)
    msgs = _sample_messages(df, n_wa=30, n_dc=30, n_gpt=10, n_claude=10)
    
    msg_block = "\n".join(f'- "{m}"' for m in msgs[:30])
    reply_sec = stats.get("avg_reply_seconds")
    reply_str = f"{reply_sec:.0f} seconds" if reply_sec else "unknown"
    
    ghosting = stats.get("ghosting_count", 0)
    ignored = stats.get("ignored_count", 0)
    monologue = stats.get("longest_monologue", 0)
    
    prompt = f"""You are a brutally honest, hilarious stand-up comedian and digital analyst. Your job is to ROAST the user based on their actual messaging data. 
    Be witty, savage, but ultimately funny. Do not hold back. Look at their actual statistics and messages to formulate your insults.

    {known_blk}

    === EMBARRASSING STATS ===
    Messages sent: {stats.get('my_messages_clean', 0):,}
    Longest streak: {stats.get('max_streak_days', 0)} days
    Night-owl activity: {stats.get('night_owl_pct', 0)}%
    Reply speed: {reply_str}
    Ghosted others: {ghosting} times
    Got ignored/left on read: {ignored} times
    Longest monologue (messages sent in a row to someone without them replying): {monologue} messages! (Yikes!)
    Swear jar count: {stats.get('swear_jar', 0)}
    Most used emojis: {" ".join(e["emoji"] for e in stats.get("top_emojis", [])[:5])}
    
    === MESSAGES THEY ACTUALLY SENT ===
    {msg_block}

    === TASK ===
    Write a brutal 3-4 paragraph roast.
    Focus on their reply speed, ghosting habits, their cringe monologue length, their excessive or weird emoji use, and their late-night texting.
    Reference their actual name if available.
    
    Output ONLY the roast text. No JSON, no markdown formatting blocks, just the pure comedic text.
    """

    try:
        if model.startswith("gemini"):
            response_text = _call_gemini(prompt)
        elif model.startswith("openai"):
            oai_model = model.split(":", 1)[1] if ":" in model else "gpt-4o-mini"
            response_text = _call_openai(prompt, oai_model)
        else:
            response_text = _call_ollama(prompt, model)
            
        return {"success": True, "roast": response_text.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Interactive Chat ──────────────────────────────────────────────────────────
def _retrieve_context(query: str, df: pd.DataFrame, max_msgs: int = 100) -> str:
    """Finds messages in df that relate to the user's query."""
    query_lower = query.lower()
    words = [w for w in re.findall(r"\b\w{3,}\b", query_lower) if w not in {"what", "who", "where", "when", "why", "how", "tell", "about", "did", "does", "is", "are", "the", "my", "me", "chat", "chats", "with"}]
    
    if not words:
        # If no specific keywords, just pull recent general messages
        sample = df[df["is_me"] == True].tail(max_msgs)
    else:
        # Score rows based on keyword matches in author, channel, or message
        def score_row(row):
            s = 0
            text = f"{row['author']} {row['channel']} {row['message']}".lower()
            for w in words:
                if w in text:
                    s += 1
            return s
        
        # We only score a recent subset to keep it fast, say last 20k msgs
        recent_df = df.tail(20000).copy()
        recent_df["_score"] = recent_df.apply(score_row, axis=1)
        
        # Take the top matches
        sample = recent_df[recent_df["_score"] > 0].sort_values(by=["_score", "timestamp"], ascending=[False, True]).head(max_msgs)
        if len(sample) == 0:
            sample = recent_df[recent_df["is_me"] == True].tail(max_msgs)
            
    # Format the messages
    lines = []
    for _, r in sample.iterrows():
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M") if pd.notnull(r["timestamp"]) else ""
        lines.append(f"[{ts}] {r['channel']} - {r['author']}: {r['message']}")
        
    return "\n".join(lines)


def chat_with_data(history: list, df: pd.DataFrame, stats: dict, metadata: dict, model: str = DEFAULT_MODEL, context_summaries: dict = None) -> dict:
    """
    Handles conversational Q&A about the user's data.
    `history` is a list of dicts: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    # 1. Get the latest user query
    latest_query = history[-1]["content"] if history else ""
    
    # 2. Retrieve relevant context
    context_msgs = _retrieve_context(latest_query, df)
    
    # 3. Build system prompt
    facts = _infer_identity(df, metadata)
    known_blk = _build_known_facts_block(facts)
    
    top_chats = ", ".join(f"{k} ({v} msgs)" for k, v in list(stats.get("top_channels", {}).items())[:5])
    
    system_prompt = f"""You are 'YourMind', an AI agent integrated into a personal chat analytics dashboard.
The user is asking you questions about their personal chat history (WhatsApp, Discord, ChatGPT, Claude).

{known_blk}

=== GENERAL STATS ===
Total msgs: {stats.get('total_messages', 0):,}
Peak activity hour: {stats.get('peak_hour', 22)}:00
Top Contacts: {top_chats}

=== RELEVANT RETRIEVED MESSAGES FOR CURRENT QUERY ===
(These messages were retrieved based on the user's recent question)
{context_msgs}

=== GENERATED CHAT SUMMARIES CONTEXT ===
{chr(10).join([f"Summary for {k}: {v}" for k, v in (context_summaries or {}).items()]) if context_summaries else "(No summaries generated yet)"}

=== INSTRUCTIONS ===
1. Answer the user's question directly, using the retrieved messages and stats as evidence.
2. Be conversational, insightful, and concise. 
3. If the user asks about a specific person, look at the retrieved messages to describe the dynamic.
4. If you don't have enough information in the retrieved messages, say so, but try to infer what you can.
5. Format your response cleanly using Markdown.
"""

    # 4. Construct API payload depending on the model
    # Convert history to prompt string for Ollama/Gemini
    conversation = []
    for msg in history[:-1]:
        role = "User" if msg["role"] == "user" else "YourMind"
        conversation.append(f"{role}: {msg['content']}")
        
    conversation.append(f"User: {latest_query}")
    conversation_text = "\n\n".join(conversation)
    
    final_prompt = f"{system_prompt}\n\n=== CONVERSATION HISTORY ===\n{conversation_text}\n\nYourMind:"
    
    try:
        # Call the appropriate model
        if model.startswith("gemini"):
            response_text = _call_gemini(final_prompt)
        elif model.startswith("openai"):
            oai_model = model.split(":", 1)[1] if ":" in model else "gpt-4o-mini"
            # For OpenAI we can pass the actual history array
            api_key = os.environ.get("OPENAI_API_KEY", "")
            messages = [{"role": "system", "content": system_prompt}] + history
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": oai_model, "messages": messages, "temperature": 0.7, "max_tokens": 2000},
                timeout=60
            )
            resp.raise_for_status()
            response_text = resp.json()["choices"][0]["message"]["content"]
        else:
            response_text = _call_ollama(final_prompt, model)
            
        return {"success": True, "response": response_text.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Single Chat Summarizer ────────────────────────────────────────────────────
def summarize_single_chat(channel: str, df: pd.DataFrame, model: str = DEFAULT_MODEL) -> dict:
    """Generate a narrative summary for a specific chat channel."""
    chat_df = df[df["channel"] == channel].copy()
    if len(chat_df) == 0:
        return {"success": False, "error": "Chat not found"}
        
    chat_df = chat_df.sort_values("timestamp")
    
    # Take the last 150 messages for context
    sample = chat_df.tail(150)
    
    lines = []
    for _, r in sample.iterrows():
        ts = r["timestamp"].strftime("%Y-%m-%d") if pd.notnull(r["timestamp"]) else ""
        lines.append(f"[{ts}] {r['author']}: {r['message']}")
        
    msg_block = "\n".join(lines)
    
    prompt = f"""You are an elite digital-behavior psychologist and relationship analyst specializing in deep dives into chat conversations.
    
    CHANNEL/CHAT NAME: {channel}
    TOTAL MESSAGES IN CHAT: {len(chat_df)}
    
    === RECENT SAMPLE MESSAGES ===
    {msg_block}
    
    === TASK ===
    Write a highly detailed, insightful, and engaging narrative summary of this chat relationship.
    You must include:
    1. The chat habits of both users (e.g., who texts more, who uses emojis, message length, response styles).
    2. What they usually talk about and the top topics covered in the sample.
    3. What they mean to each other (e.g., are they best friends, colleagues, partners, casual acquaintances?) and the dynamic of their relationship.
    4. How their personalities match or clash.
    5. The overall vibe and tone of the conversation.
    6. A "Wildcard Insight" - a unique, brainstormed psychological observation or fun deduction about them based on the text.
    
    Format this as a readable, engaging narrative (3-4 paragraphs). Be bold, specific, and entertaining. Reference the actual sample messages provided.
    Do not use JSON or markdown code blocks, just return the plain text summary (you can use basic markdown like bold text).
    """

    try:
        if model.startswith("gemini"):
            response_text = _call_gemini(prompt)
        elif model.startswith("openai"):
            oai_model = model.split(":", 1)[1] if ":" in model else "gpt-4o-mini"
            response_text = _call_openai(prompt, oai_model)
        else:
            response_text = _call_ollama(prompt, model)
            
        return {"success": True, "summary": response_text.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}
