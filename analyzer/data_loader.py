"""
data_loader.py — loads all chat exports into a unified DataFrame.
Sources: WhatsApp (.txt), Discord (data package), ChatGPT (JSON), Claude (JSON)

Identity is auto-detected per platform rather than hardcoded.
"""

import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

import os
BASE_PATH = Path(os.environ.get("DATA_EXPORT_PATH", "D:/DataExport"))


# ── Identity auto-detection ──────────────────────────────────────────────────
def _detect_me_wa(df_wa: pd.DataFrame) -> str:
    """Most messages in WA usually come from you in your own export."""
    counts = df_wa["author"].value_counts()
    return counts.index[0] if len(counts) else ""


def _detect_me_dc(df_dc: pd.DataFrame) -> str:
    """Most sent Discord messages = your account."""
    counts = df_dc["author"].value_counts()
    return counts.index[0] if len(counts) else ""


# ── Text extractors ──────────────────────────────────────────────────────────
def _extract_gpt_text(content) -> str:
    if isinstance(content, dict):
        parts = content.get("parts", [])
        return " ".join(p for p in parts if isinstance(p, str))
    if isinstance(content, str):
        return content
    return ""


def _extract_claude_text(msg: dict) -> str:
    """
    Claude export stores text in the top-level 'text' field on human messages.
    Content blocks are often empty stubs. Prefer top-level text, fall back to blocks.
    """
    # Top-level text (most reliable for human messages)
    top = str(msg.get("text", "")).strip()
    if top:
        return top

    # Content blocks (assistant responses are usually here)
    content = msg.get("content", [])
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                t = str(b.get("text", "")).strip()
                if t:
                    parts.append(t)
        if parts:
            return " ".join(parts)

    return ""


# ── Channel name cleaner (generic) ──────────────────────────────────────────
_LONG_WORD_RE = re.compile(r'\b\w{20,}\b')   # strip numeric Discord snowflakes etc.

def _clean_channel(name: str, source: str) -> str:
    """Produce a readable display name for a channel / conversation."""
    name = str(name).strip()

    if source == "WhatsApp":
        name = re.sub(r"^WhatsApp Chat with ", "", name)

    if source == "Discord":
        # Remove Discord numeric IDs in brackets: [744238584592203877]
        name = re.sub(r"\s*\[\d+\]$", "", name)
        # Remove "Direct Messages - " prefix
        name = re.sub(r"^Direct Messages - ", "", name)
        # Handle "Server Name - #channel" pattern:
        # If there's a " - " separator and the left side is long (>20 chars),
        # abbreviate the left side to its initials/first word, keep the channel part.
        if " - " in name:
            parts = name.split(" - ", 1)
            server, channel = parts[0].strip(), parts[1].strip()
            if len(server) > 15:
                    # Acronym: first letter of each word
                    words = re.findall(r"[A-Za-z]+", server)
                    acronym = "".join(w[0].upper() for w in words if w)
                    if len(acronym) < 2:
                        acronym = re.sub(r"[^A-Za-z0-9]", "", server)[:6]
                    name = f"{acronym}: {channel}"
        # Final cap at 35 chars
        if len(name) > 35:
            name = name[:33].rstrip() + "…"

    return name.strip()



# ── WhatsApp Parser ──────────────────────────────────────────────────────────
def _parse_whatsapp_file(file_path: Path):
    """Parses a raw WhatsApp .txt export."""
    rows = []
    # Patterns: 
    # [15/01/21, 10:44:52] Author: Message
    # 15/01/21, 10:44 - Author: Message
    # 2021-01-15, 10:44:52 - Author: Message
    pattern = re.compile(r'^\[?(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4},?\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\]?[\s-]*([^:]+):\s(.*)$')
    
    channel_name = _clean_channel(file_path.stem, "WhatsApp")
    
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            current_msg = None
            for line in f:
                line = line.strip()
                if not line: continue
                
                match = pattern.match(line)
                if match:
                    ts_str, author, text = match.groups()
                    current_msg = {
                        "timestamp": ts_str,
                        "author": author.strip(),
                        "channel": channel_name,
                        "message": text.strip(),
                        "source": "WhatsApp",
                        "is_me": False # placeholder, updated later
                    }
                    rows.append(current_msg)
                elif current_msg:
                    # Multiline message
                    current_msg["message"] += " " + line
    except Exception as e:
        print(f"[ERROR] Failed to parse WhatsApp file {file_path}: {e}")
        
    return rows


# ── Discord Parser ───────────────────────────────────────────────────────────
def _parse_discord_data(discord_root: Path):
    """Parses Discord chat exports (JSON files)."""
    rows = []
    
    # Scan for .json files in the DISCORD directory
    for f in discord_root.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as file:
                data = json.load(file)
                
            ch_info = data.get("channel", {})
            channel_name = _clean_channel(ch_info.get("name") or f.stem, "Discord")
            
            msgs = data.get("messages", [])
            for m in msgs:
                author_info = m.get("author", {})
                author_name = author_info.get("name") or author_info.get("nickname") or "Unknown"
                
                rows.append({
                    "timestamp": m.get("timestamp"),
                    "author":    author_name,
                    "channel":   channel_name,
                    "message":   str(m.get("content", "")),
                    "source":    "Discord",
                    "is_me":     False # detected later
                })
        except Exception as e:
            print(f"[ERROR] Failed to parse Discord JSON {f.name}: {e}")
            
    return rows


# ── Main loader ──────────────────────────────────────────────────────────────
def load_all_data():
    all_rows = []
    metadata = {
        "gpt_titles":    [],
        "claude_topics": [],
        "my_wa_name":    "",
        "my_dc_name":    "",
    }

    # ── 1. WhatsApp ──────────────────────────────────────────────────────────
    wa_dir = BASE_PATH / "WHATSAPP"
    if wa_dir.exists():
        print(f"[*] Found WhatsApp directory: {wa_dir}")
        for f in wa_dir.glob("*.txt"):
            all_rows.extend(_parse_whatsapp_file(f))

    # ── 2. Discord ───────────────────────────────────────────────────────────
    dc_dir = BASE_PATH / "DISCORD"
    if dc_dir.exists():
        print(f"[*] Found Discord directory: {dc_dir}")
        all_rows.extend(_parse_discord_data(dc_dir))

    # ── 3. ChatGPT ───────────────────────────────────────────────────────────
    gpt_path = BASE_PATH / "GPT" / "conversations.json"
    if gpt_path.exists():
        with open(gpt_path, encoding="utf-8") as f:
            gpt_data = json.load(f)

        for conv in gpt_data:
            title   = (conv.get("title") or "Untitled").strip()
            if title and title.lower() != "untitled":
                metadata["gpt_titles"].append(title)
            create_ts = conv.get("create_time")
            channel   = f"GPT: {title[:55]}"

            for node in conv.get("mapping", {}).values():
                msg = node.get("message")
                if not msg:
                    continue
                role = msg.get("author", {}).get("role", "")
                if role not in ("user", "assistant"):
                    continue

                text = _extract_gpt_text(msg.get("content", {}))
                if not text.strip():
                    continue

                raw_ts = msg.get("create_time") or create_ts
                try:
                    dt = datetime.utcfromtimestamp(float(raw_ts)).isoformat() + "Z" if raw_ts else None
                except Exception:
                    dt = None

                all_rows.append({
                    "timestamp": dt,
                    "author":    "me" if role == "user" else "ChatGPT",
                    "channel":   channel,
                    "message":   text,
                    "source":    "ChatGPT",
                    "is_me":     role == "user",
                })

    # ── 3. Claude ────────────────────────────────────────────────────────────
    claude_root = BASE_PATH / "CLAUDE"
    if claude_root.exists():
        for entry in claude_root.iterdir():
            # Support both: batch subdirectories AND conversations.json directly inside CLAUDE/
            if entry.is_dir():
                conv_path = entry / "conversations.json"
            elif entry.name == "conversations.json":
                conv_path = entry
            else:
                continue

            if not conv_path.exists():
                continue

            with open(conv_path, encoding="utf-8") as f:
                try:
                    claude_data = json.load(f)
                except Exception:
                    continue

            if not isinstance(claude_data, list):
                claude_data = [claude_data]

            for conv in claude_data:
                name    = (conv.get("name") or "Untitled").strip()
                if name and name.lower() != "untitled":
                    metadata["claude_topics"].append(name)
                channel = f"Claude: {name[:55]}"

                for msg in conv.get("chat_messages", []):
                    sender = msg.get("sender", "")
                    if sender not in ("human", "assistant"):
                        continue

                    text = _extract_claude_text(msg)
                    if not text.strip():
                        continue

                    raw_ts = msg.get("created_at", "")
                    try:
                        dt = datetime.fromisoformat(
                            raw_ts.replace("Z", "+00:00")
                        ).isoformat() if raw_ts else None
                    except Exception:
                        dt = raw_ts or None

                    all_rows.append({
                        "timestamp": dt,
                        "author":    "me" if sender == "human" else "Claude",
                        "channel":   channel,
                        "message":   text,
                        "source":    "Claude",
                        "is_me":     sender == "human",
                    })

    if not all_rows:
        df = pd.DataFrame(columns=["timestamp", "author", "channel", "message", "source", "is_me"])
        return df, metadata

    df = pd.DataFrame(all_rows)
    
    # Normalize source names
    def _norm_src(s):
        s = str(s).lower().strip()
        if s == "whatsapp": return "WhatsApp"
        if s == "discord":  return "Discord"
        if s == "chatgpt":  return "ChatGPT"
        if s == "claude":   return "Claude"
        return s.title()
    df["source"] = df["source"].apply(_norm_src)
    
    # Process identities
    wa_mask = df["source"].str.lower() == "whatsapp"
    df_wa = df[wa_mask]
    if not df_wa.empty:
        my_wa = _detect_me_wa(df_wa)
        metadata["my_wa_name"] = my_wa
        df.loc[wa_mask, "is_me"] = df["author"] == my_wa

    dc_mask = df["source"].str.lower() == "discord"
    df_dc = df[dc_mask]
    if not df_dc.empty:
        my_dc = _detect_me_dc(df_dc)
        metadata["my_dc_name"] = my_dc
        df.loc[dc_mask, "is_me"] = df["author"] == my_dc
    
    # Process timestamps correctly based on source
    wa_mask = df["source"].str.lower() == "whatsapp"
    df.loc[wa_mask, "timestamp"] = pd.to_datetime(df.loc[wa_mask, "timestamp"], errors="coerce", format="mixed").dt.tz_localize(None)
    
    other_mask = ~wa_mask
    local_tz = datetime.now().astimezone().tzinfo
    df.loc[other_mask, "timestamp"] = pd.to_datetime(df.loc[other_mask, "timestamp"], errors="coerce", format="mixed", utc=True).dt.tz_convert(local_tz).dt.tz_localize(None)

    # Force the entire column to be datetime dtype
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp", "message"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Clean up messages
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0].reset_index(drop=True)

    return df, metadata
