"""
stats.py — compute every statistic from the unified DataFrame.
Includes spam-aware deduplication for accurate word / chat / emoji analysis.
"""

import re
import emoji
from collections import Counter
import pandas as pd
import numpy as np
from pathlib import Path

URL_RE = re.compile(r"http\S+|www\S+")

STOP_WORDS = {
    "i","the","a","an","is","it","to","of","and","in","that","my","you","me","we",
    "are","was","be","do","for","on","have","with","but","not","so","like","just",
    "at","this","or","he","she","they","his","her","your","our","its","will","can",
    "would","could","should","about","up","if","no","what","how","all","yeah","ok",
    "okay","oh","yea","yes","omitted","media","image","video","audio","sticker",
    "gif","file","attached","im","dont","its","ive","id","ill","were","thats",
    "also","been","get","got","one","then","when","from","had","has","him","than",
    "even","know","think","lol","lmao","bro","man","wait","already","because","there",
    "their","which","who","more","some","now","here","going","want","much","really",
    "actually","then","thing","make","see","time","back","still","come","look","gonna",
    "wanna","need","use","way","day","good","great","nice","sure","right","well","haha",
    "hi","hey","hello","bye","send","sent","said","tell","told","go","say","see",
    "did","does","them","these","those","am","only","other","same","being",
    "haat","ha","haha","hahaha","hahahaha","ok","okay","lmfao","lmao","lol",
    # code/technical noise (from GPT/Claude transcripts)
    "nan","int","string","class","new","return","false","true","null","void",
    "def","var","let","const","function","import","print","println","system",
    "deleted","message","error","none","type","list","dict","str","arr",
    "output","input","data","code","file","user","api","text","result",
    "value","key","object","model","response","request","server","client",
}

SWEAR_WORDS = {
    "fuck", "fucking", "fucked", "shit", "shitty", "bitch", "ass", "asshole", 
    "damn", "crap", "bastard", "dick", "cunt", "slut", "whore", "pussy"
}

VIBE_WORDS_POS = {
    "love", "amazing", "great", "awesome", "good", "happy", "best", "perfect",
    "beautiful", "nice", "fun", "excited", "yay", "cool", "sweet", "thanks", "thank",
    "proud", "brilliant", "fantastic", "wonderful", "glad", "blessed", "vibes"
}

VIBE_WORDS_NEG = {
    "bad", "terrible", "awful", "sad", "hate", "worst", "angry", "annoying",
    "stupid", "dumb", "ugly", "mad", "upset", "depressed", "tired", "exhausted",
    "sick", "hurt", "pain", "boring", "sucks", "fucking", "shit", "fuck", "damn"
}

# ── Spam Deduplication ───────────────────────────────────────────────────────
def _deduplicate(df: pd.DataFrame,
                 window_seconds: int = 120,
                 max_repeats: int = 2) -> pd.DataFrame:
    """
    Within each (channel, author) group, if the same message (case-insensitive,
    stripped) repeats more than `max_repeats` times inside a rolling `window_seconds`
    window, keep only the first `max_repeats` occurrences.

    This kills spam bursts (e.g. "haat" sent 80× in a minute) without removing
    legitimate repeated phrases.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["_msg_key"] = df["message"].str.strip().str.lower()

    keep_mask = pd.Series(True, index=df.index)

    for (channel, author), grp in df.groupby(["channel", "author"], sort=False):
        grp = grp.sort_values("timestamp")
        # track how many times each msg has appeared recently
        recent: dict = {}   # msg_key -> list of timestamps

        for idx, row in grp.iterrows():
            key = row["_msg_key"]
            ts  = row["timestamp"]
            # prune old entries outside window
            if key in recent:
                recent[key] = [t for t in recent[key]
                                if (ts - t).total_seconds() <= window_seconds]
            else:
                recent[key] = []

            if len(recent[key]) >= max_repeats:
                keep_mask[idx] = False   # spam – drop
            else:
                recent[key].append(ts)

    return df[keep_mask].drop(columns=["_msg_key"])


def _extract_emojis(text: str) -> list:
    return [c for c in text if c in emoji.EMOJI_DATA]


# ── Main ────────────────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame, metadata: dict) -> dict:
    if len(df) == 0:
        return {
            "total_messages": 0, "total_messages_clean": 0, "my_messages": 0, "my_messages_clean": 0,
            "spam_removed": 0, "unique_contacts": 0, "unique_chats": 0,
            "date_range": {"start": "N/A", "end": "N/A", "days": 0},
            "avg_msg_length": 0, "avg_word_count": 0, "median_msg_length": 0, "total_words_sent": 0,
            "hourly_activity": {}, "peak_hour": 0, "night_owl_pct": 0, "late_night_pct": 0, "late_night_messages": 0,
            "estimated_sleep_hour": 0, "daily_activity": {}, "most_active_day": "N/A", "least_active_day": "N/A",
            "heatmap": [], "monthly_activity": {}, "platform_breakdown": {}, "my_platform_breakdown": {},
            "top_contacts": {}, "top_channels": {}, "top_channels_days": {}, "top_emojis": [],
            "total_emojis_sent": 0, "emoji_per_message": 0, "swear_jar": 0, "questions_asked": 0, "questions_pct": 0,
            "top_words": [], "all_text_sample": "", "max_streak_days": 0, "gpt_conversations": 0,
            "claude_conversations": 0, "gpt_titles": [], "claude_topics": [], "msg_length_dist": {},
            "avg_reply_seconds": 0, "ghosting_count": 0, "ignored_count": 0,
            "longest_monologue": 0, "time_capsule": None, "sentiment_timeline": []
        }

    stats: dict = {}

    # Raw me-df (for timing/streak stats that should NOT be deduped)
    me_raw = df[df["is_me"] == True].copy()
    me_raw["msg_len"]    = me_raw["message"].str.len()
    me_raw["word_count"] = me_raw["message"].str.split().str.len()
    me_raw["hour"]       = me_raw["timestamp"].dt.hour
    me_raw["dow"]        = me_raw["timestamp"].dt.dayofweek
    me_raw["date"]       = me_raw["timestamp"].dt.date
    me_raw["ym"]         = me_raw["timestamp"].dt.to_period("M").astype(str)

    # Deduplicated full df + me-df (for word/emoji/chat counts)
    df_clean   = _deduplicate(df,    window_seconds=120, max_repeats=2)
    me_clean   = df_clean[df_clean["is_me"] == True].copy()
    me_clean["ym"] = me_clean["timestamp"].dt.to_period("M").astype(str)

    total_raw   = max(len(me_raw), 1)
    total_clean = max(len(me_clean), 1)

    # ── Basic ────────────────────────────────────────────────────────────────
    stats["total_messages"]        = len(df)
    stats["total_messages_clean"]  = len(df_clean)
    stats["my_messages"]           = len(me_raw)
    stats["my_messages_clean"]     = len(me_clean)
    stats["spam_removed"]          = len(df) - len(df_clean)
    stats["unique_contacts"]       = df[~df["is_me"]]["author"].nunique()
    stats["unique_chats"]          = df["channel"].nunique()

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    stats["date_range"] = {
        "start": ts_min.strftime("%b %d, %Y"),
        "end":   ts_max.strftime("%b %d, %Y"),
        "days":  (ts_max - ts_min).days,
    }

    # ── Message length (raw – not affected by spam much) ─────────────────────
    stats["avg_msg_length"]    = round(float(me_raw["msg_len"].mean()), 1)
    stats["avg_word_count"]    = round(float(me_raw["word_count"].mean()), 1)
    stats["median_msg_length"] = round(float(me_raw["msg_len"].median()), 1)
    stats["total_words_sent"]  = int(me_raw["word_count"].sum())

    # ── Hourly activity (raw – timing is real) ───────────────────────────────
    hourly = me_raw.groupby("hour").size()
    stats["hourly_activity"] = {int(h): int(c) for h, c in hourly.items()}
    stats["peak_hour"]        = int(hourly.idxmax()) if len(hourly) else 0

    night = me_raw[me_raw["hour"].between(20, 23)]
    late  = me_raw[me_raw["hour"].between(0,  4)]
    stats["night_owl_pct"]       = round(len(night) / total_raw * 100, 1)
    stats["late_night_pct"]      = round(len(late)  / total_raw * 100, 1)
    stats["late_night_messages"] = len(late)

    sleep_window = hourly.reindex(range(9), fill_value=0)
    stats["estimated_sleep_hour"] = int(sleep_window.idxmin())

    # ── Day-of-week activity (raw) ────────────────────────────────────────────
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = me_raw.groupby("dow").size().reindex(range(7), fill_value=0)
    stats["daily_activity"]   = {dow_names[i]: int(v) for i, v in dow.items()}
    stats["most_active_day"]  = dow_names[int(dow.idxmax())]
    stats["least_active_day"] = dow_names[int(dow.idxmin())]

    # ── Activity heatmap (raw) ────────────────────────────────────────────────
    hm = me_raw.groupby(["dow", "hour"]).size().reset_index(name="count")
    stats["heatmap"] = hm.to_dict("records")

    # ── Monthly trend (raw) ──────────────────────────────────────────────────
    monthly = me_raw.groupby("ym").size()
    stats["monthly_activity"] = {k: int(v) for k, v in monthly.items()}

    # ── Platform breakdown ───────────────────────────────────────────────────
    stats["platform_breakdown"]    = {k: int(v) for k, v in df["source"].value_counts().items()}
    stats["my_platform_breakdown"] = {k: int(v) for k, v in me_raw["source"].value_counts().items()}

    # ── TOP CONTACTS — spam-deduped; AI sources excluded ────────────────────
    # Determine bot author names dynamically (authors from AI sources)
    ai_sources = {"ChatGPT", "Claude"}
    ai_authors  = set(df_clean[df_clean["source"].isin(ai_sources)]["author"].unique())
    non_me_clean = df_clean[~df_clean["is_me"] & ~df_clean["author"].isin(ai_authors)]
    top_contacts_raw = non_me_clean["author"].value_counts().head(15)
    stats["top_contacts"] = {k: int(v) for k, v in top_contacts_raw.items()}

    # ── TOP CHANNELS — engagement-weighted score ─────────────────────────────
    # Score = unique_days^1.5 * log10(msg_count + 1)
    # This rewards long-running, spread-out conversations over spam bursts.
    # Uses spam-cleaned message counts. Social platforms only (WA + Discord).
    is_social = df_clean["source"].str.lower().isin(["whatsapp", "discord"])
    social_clean = me_clean[me_clean["source"].str.lower().isin(["whatsapp", "discord"])].copy()
    social_clean["date"] = social_clean["timestamp"].dt.date

    ch_msgs = social_clean.groupby("channel").size().rename("msg_count")
    ch_days = social_clean.groupby("channel")["date"].nunique().rename("unique_days")
    ch_score = ((ch_days ** 1.5) * np.log10(ch_msgs + 1)).rename("score")

    top_ch = (
        pd.concat([ch_msgs, ch_days, ch_score], axis=1)
        .sort_values(["score", "unique_days", "msg_count"], ascending=[False, False, False])
        .head(12)
    )
    stats["top_channels"] = {
        ch: int(row["msg_count"]) for ch, row in top_ch.iterrows()
    }
    stats["top_channels_days"] = {
        ch: int(row["unique_days"]) for ch, row in top_ch.iterrows()
    }

    # ── EMOJIS AND SURPRISE STATS (Swear Jar & Questions) ───────────────────
    all_emojis: list = []
    swear_count = 0
    questions_count = 0
    
    for msg in me_clean["message"]:
        m_str = str(msg).lower()
        all_emojis.extend(_extract_emojis(m_str))
        
        # Swear jar
        words = re.findall(r"\b[a-z]{2,}\b", m_str)
        for w in words:
            if w in SWEAR_WORDS:
                swear_count += 1
                
        # Questions
        if "?" in m_str:
            questions_count += 1
            
    emoji_counts = Counter(all_emojis).most_common(20)
    stats["top_emojis"]        = [{"emoji": e, "count": int(c)} for e, c in emoji_counts]
    stats["total_emojis_sent"] = len(all_emojis)
    stats["emoji_per_message"] = round(len(all_emojis) / total_clean, 3)
    stats["swear_jar"] = swear_count
    
    stats["questions_asked"] = questions_count
    stats["questions_pct"]   = round(questions_count / total_clean * 100, 1)

    # Word frequency — only social (WA + Discord) to avoid GPT/Claude code pollution
    word_source_df = me_clean[me_clean["source"].str.lower().isin(["whatsapp", "discord"])]
    all_words: list = []
    for msg in word_source_df["message"]:
        clean = URL_RE.sub("", str(msg).lower())
        words = re.findall(r"\b[a-z]{3,}\b", clean)
        all_words.extend(w for w in words if w not in STOP_WORDS)
    word_counts = Counter(all_words).most_common(60)
    stats["top_words"]       = [{"word": w, "count": int(c)} for w, c in word_counts]
    
    sample_text = " ".join(all_words[:50_000])
    stats["all_text_sample"] = sample_text
    print(f"[*] Generated word cloud text sample: {len(sample_text)} characters.")

    # ── Streak (raw – timing) ─────────────────────────────────────────────────
    dates = sorted(me_raw["date"].unique())
    max_streak = cur = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days == 1:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 1
    stats["max_streak_days"] = max_streak

    # ── GPT / Claude metadata ─────────────────────────────────────────────────
    stats["gpt_conversations"]    = df[df["source"] == "ChatGPT"]["channel"].nunique()
    stats["claude_conversations"] = df[df["source"] == "Claude"]["channel"].nunique()
    stats["gpt_titles"]           = metadata.get("gpt_titles", [])
    stats["claude_topics"]        = metadata.get("claude_topics", [])

    # ── Message length distribution (deduped) ────────────────────────────────
    me_clean2 = me_clean.copy()
    me_clean2["msg_len"] = me_clean2["message"].str.len()
    bins   = [0, 10, 30, 80, 200, 500, 99999]
    labels = ["1-10", "11-30", "31-80", "81-200", "201-500", "500+"]
    me_clean2["len_bucket"] = pd.cut(me_clean2["msg_len"], bins=bins, labels=labels)
    len_dist = me_clean2["len_bucket"].value_counts().sort_index()
    stats["msg_length_dist"] = {str(k): int(v) for k, v in len_dist.items()}

    # ── Questions asked (deduped) ─────────────────────────────────────────────
    q_count = me_clean["message"].str.count(r"\?").sum()
    stats["questions_asked"] = int(q_count)
    stats["questions_pct"]   = round(float(q_count) / total_clean * 100, 1)

    # ── Reply speed: avg seconds between incoming → my reply (WA/DC only) ────
    social_df = df_clean[df_clean["source"].isin(["WhatsApp", "Discord"])].copy()
    reply_gaps = []
    for ch, grp in social_df.groupby("channel"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        for i in range(1, len(grp)):
            if grp.loc[i, "is_me"] and not grp.loc[i-1, "is_me"]:
                gap = (grp.loc[i, "timestamp"] - grp.loc[i-1, "timestamp"]).total_seconds()
                if 0 < gap < 3600:   # ignore gaps > 1 hour
                    reply_gaps.append(gap)
    stats["avg_reply_seconds"] = round(float(np.median(reply_gaps)), 0) if reply_gaps else None

    # ── Relational Metrics (Ghosting & Monologues) ────────────────────────────
    ghosting_count = 0
    ignored_count = 0
    longest_monologue = 0
    
    for ch, grp in social_df.groupby("channel"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        current_monologue = 0
        
        for i in range(1, len(grp)):
            prev_me = grp.loc[i-1, "is_me"]
            curr_me = grp.loc[i, "is_me"]
            gap = (grp.loc[i, "timestamp"] - grp.loc[i-1, "timestamp"]).total_seconds()
            
            # Monologue tracking
            if curr_me and prev_me:
                current_monologue += 1
                longest_monologue = max(longest_monologue, current_monologue)
            elif curr_me and not prev_me:
                current_monologue = 1
                
            # Ghosting (I got a message, didn't reply for > 24 hours or ever)
            # If gap > 24 hours, whoever didn't reply ghosted.
            if gap > 86400: # 24 hours
                if not prev_me and curr_me:
                    # They sent a message, I replied 24 hours later -> I ghosted them
                    ghosting_count += 1
                elif prev_me and not curr_me:
                    # I sent a message, they replied 24 hours later -> They ghosted me (ignored)
                    ignored_count += 1
                    
    stats["ghosting_count"] = ghosting_count
    stats["ignored_count"] = ignored_count
    stats["longest_monologue"] = longest_monologue

    # ── Sentiment & Vibe Timeline ─────────────────────────────────────────────
    # Calculate rolling sentiment based on positive/negative word hits in my messages
    sentiment_data = []
    
    # We will group by month-year for the timeline
    for ym, group in me_clean.groupby("ym"):
        pos_hits = 0
        neg_hits = 0
        total_words = 0
        for msg in group["message"]:
            m_str = str(msg).lower()
            words = re.findall(r"\b[a-z]{2,}\b", m_str)
            total_words += len(words)
            for w in words:
                if w in VIBE_WORDS_POS: pos_hits += 1
                if w in VIBE_WORDS_NEG: neg_hits += 1
        
        # very simple sentiment score: (pos - neg) / (pos + neg + 1) * 100
        if (pos_hits + neg_hits) > 0:
            score = round((pos_hits - neg_hits) / (pos_hits + neg_hits) * 100, 1)
        else:
            score = 0
        sentiment_data.append({"month": ym, "score": score})
    
    # sort by month chronologically
    sentiment_data = sorted(sentiment_data, key=lambda x: x["month"])
    stats["sentiment_timeline"] = sentiment_data

    # ── Time Capsule ("On This Day") ──────────────────────────────────────────
    # Find a message sent by me around 1 year ago (365 days +/- 7 days)
    if not me_clean.empty:
        last_date = me_clean["timestamp"].max()
        one_year_ago = last_date - pd.Timedelta(days=365)
        
        # find messages within +/- 7 days of 1 year ago
        capsule_candidates = me_clean[
            (me_clean["timestamp"] >= one_year_ago - pd.Timedelta(days=7)) &
            (me_clean["timestamp"] <= one_year_ago + pd.Timedelta(days=7)) &
            (me_clean["message"].str.len() > 15) & # substantive message
            (~me_clean["message"].str.contains("http", case=False, na=False)) # no links
        ]
        
        if not capsule_candidates.empty:
            # pick a random one
            picked = capsule_candidates.sample(1).iloc[0]
            
            # Brainstorming some personality-based insight for the message
            msg_text = picked["message"]
            msg_len = len(msg_text)
            hour = picked["timestamp"].hour
            
            insight = "Just a typical day."
            if hour >= 23 or hour < 4:
                insight = "A late-night thought from exactly a year ago. Sleep deprivation really brings out the deepest (or weirdest) side of you."
            elif msg_len > 200:
                insight = "You were definitely feeling talkative exactly a year ago. Look at this massive paragraph you dropped."
            elif len(_extract_emojis(msg_text)) > 3:
                insight = "You were clearly very expressive exactly a year ago. The emoji game was strong."
            elif any(w in str(msg_text).lower() for w in VIBE_WORDS_NEG):
                insight = "You seemed a bit stressed or annoyed exactly a year ago. Hopefully today is better!"
            elif any(w in str(msg_text).lower() for w in VIBE_WORDS_POS):
                insight = "A positive vibe from exactly a year ago. Keep that same energy today!"
            elif "?" in msg_text:
                insight = "You were seeking answers exactly a year ago. Did you ever figure it out?"
            else:
                insight = "A blast from the past. Time flies, doesn't it?"

            stats["time_capsule"] = {
                "date": picked["timestamp"].strftime("%b %d, %Y"),
                "channel": picked["channel"],
                "message": msg_text,
                "insight": insight
            }
        else:
            stats["time_capsule"] = None
    else:
        stats["time_capsule"] = None

    # ── Export Filtered Data ──────────────────────────────────────────────────
    try:
        import os
        export_dir = Path(__file__).parent.parent / "filtered_data"
        os.makedirs(export_dir, exist_ok=True)
        
        # Save combined
        df_clean.to_csv(export_dir / "all_clean_messages.csv", index=False)
        # Save per source
        for src in df_clean["source"].unique():
            src_df = df_clean[df_clean["source"] == src]
            safe_src = str(src).lower().replace(" ", "_")
            src_df.to_csv(export_dir / f"{safe_src}_clean.csv", index=False)
    except Exception as e:
        print(f"[ERROR] Failed to export filtered data: {e}")

    return stats


def get_chat_summaries(df: pd.DataFrame) -> dict:
    """
    Computes lightweight meta-summaries for all chats grouped by platform.
    """
    if len(df) == 0:
        return {}
        
    df_clean = _deduplicate(df, window_seconds=120, max_repeats=2)
    results = {}
    
    for source in df_clean["source"].unique():
        results[source] = []
        source_df = df_clean[df_clean["source"] == source]
        
        for channel, grp in source_df.groupby("channel"):
            msg_count = len(grp)
            start_date = grp["timestamp"].min().strftime("%b %Y") if pd.notnull(grp["timestamp"].min()) else ""
            end_date = grp["timestamp"].max().strftime("%b %Y") if pd.notnull(grp["timestamp"].max()) else ""
            
            participants = grp["author"].value_counts().head(3).index.tolist()
            
            pos_hits = 0
            neg_hits = 0
            all_words = []
            
            # Sample for speed
            sample_msgs = grp["message"].dropna().tail(300)
            for msg in sample_msgs:
                m_str = str(msg).lower()
                clean = URL_RE.sub("", m_str)
                words = re.findall(r"\b[a-z]{3,}\b", clean)
                for w in words:
                    if w in VIBE_WORDS_POS: pos_hits += 1
                    if w in VIBE_WORDS_NEG: neg_hits += 1
                    if w not in STOP_WORDS: all_words.append(w)
                    
            if pos_hits > neg_hits * 1.5:
                vibe = "Positive ✨"
            elif neg_hits > pos_hits * 1.5:
                vibe = "Negative 🌧️"
            else:
                vibe = "Neutral ⚖️"
                
            top_keywords = [w for w, c in Counter(all_words).most_common(4)]
            
            results[source].append({
                "channel": channel,
                "msg_count": msg_count,
                "start": start_date,
                "end": end_date,
                "participants": participants,
                "vibe": vibe,
                "keywords": top_keywords
            })
            
        results[source].sort(key=lambda x: x["msg_count"], reverse=True)
        results[source] = results[source][:40] # Keep top 40 per platform
        
    return results
