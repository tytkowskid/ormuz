"""
Sentiment analyst agent.

Both data sources for 2019 are large (the TSLA tweet file alone is ~285k
rows / 46 MB), so the tools never hand raw rows to the model. Instead they:

  1. clamp the requested window to at most MAX_LOOKBACK_DAYS before end_date,
  2. score each item's polarity,
  3. roll the polarity up into a per-day table, and
  4. attach only a handful of representative items verbatim.

News polarity comes straight from Alpha Vantage's per-ticker
`sentiment_score`. Tweets have no score, so a small bull/bear keyword
lexicon assigns each tweet -1 / 0 / +1 (see `_tweet_polarity`).
"""

from pathlib import Path
import ast
import re

import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_DIR = Path("data")
NEWS_DIR = DATA_DIR / "news"
TWITTER_DIR = DATA_DIR / "twitter"

MAX_LOOKBACK_DAYS = 14      # hard cap on the requested window
MAX_NEWS_HEADLINES = 30
MAX_SAMPLE_TWEETS = 6
TWEET_BODY_CHARS = 220

# Bull / bear keyword lexicon for dependency-free tweet polarity.
_BULL_WORDS = {
    "buy", "buying", "bought", "long", "longs", "bull", "bullish", "moon",
    "rocket", "calls", "call", "breakout", "rally", "rallying", "undervalued",
    "upgrade", "upgraded", "beat", "beats", "accumulate", "squeeze", "support",
    "strong", "strength", "surge", "surging", "outperform", "growth", "record",
    "hodl", "hold", "holding", "pump", "green", "gains",
}
_BEAR_WORDS = {
    "sell", "selling", "sold", "short", "shorts", "bear", "bearish", "crash",
    "crashing", "dump", "dumping", "puts", "put", "overvalued", "downgrade",
    "downgraded", "miss", "misses", "fraud", "scam", "fud", "bankrupt",
    "bankruptcy", "lawsuit", "sec", "investigation", "recall", "burn",
    "burning", "debt", "dilution", "resistance", "weak", "weakness", "drop",
    "dropping", "plunge", "plunging", "tank", "tanking", "red", "losses",
}
_WORD_RE = re.compile(r"[a-z]+")


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _clamp_window(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Bound [start, end] so it is never wider than MAX_LOOKBACK_DAYS."""
    end_dt = pd.to_datetime(end_date)
    start_dt = pd.to_datetime(start_date)
    floor = end_dt - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
    return max(start_dt, floor), end_dt


def _load_window(path: Path, date_col: str, start_date, end_date) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    start_dt, end_dt = _clamp_window(start_date, end_date)
    df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)].copy()
    return df.sort_values(date_col)


def _ticker_news_score(raw: object, ticker: str) -> float | None:
    """Extract this ticker's sentiment_score from the `ticker_sentiment` cell."""
    try:
        for item in ast.literal_eval(str(raw)):
            if str(item.get("ticker", "")).upper() == ticker:
                return float(item["sentiment_score"])
    except Exception:
        pass
    return None


def _tweet_polarity(text: str) -> int:
    """-1 / 0 / +1 from a bull/bear keyword count (plus a couple of emoji)."""
    s = str(text).lower()
    toks = set(_WORD_RE.findall(s))
    bull = len(toks & _BULL_WORDS) + s.count("🚀") + s.count("📈")
    bear = len(toks & _BEAR_WORDS) + s.count("📉") + s.count("🐻")
    if bull > bear:
        return 1
    if bear > bull:
        return -1
    return 0


# -------------------------------------------------------------------
# NEWS TOOL
# -------------------------------------------------------------------

@tool
def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """
    Aggregated news sentiment for `ticker` between start_date and end_date
    (YYYY-MM-DD). The window is capped at the last 14 days before end_date.

    Returns a per-day sentiment summary (article count, mean score) followed
    by the headlines only — never the full article text. Score sign: > +0.15
    bullish, < -0.15 bearish, in between neutral.
    """
    ticker = ticker.upper()
    df = _load_window(
        NEWS_DIR / f"{ticker}_news_2019.csv", "time_published", start_date, end_date
    )
    if df.empty:
        return f"No news found for {ticker} in the requested window."

    ticker_score = pd.to_numeric(
        df["ticker_sentiment"].map(lambda cell: _ticker_news_score(cell, ticker)),
        errors="coerce",
    )
    overall_score = pd.to_numeric(df["overall_sentiment_score"], errors="coerce")
    df["score"] = ticker_score.fillna(overall_score).fillna(0.0)
    df["day"] = df["time_published"].dt.date

    daily = df.groupby("day").agg(
        articles=("title", "size"),
        mean_score=("score", "mean"),
    ).reset_index()

    lines = [
        f"NEWS SENTIMENT — {ticker} — {df['day'].min()} to {df['day'].max()}",
        f"{len(df)} articles | mean ticker sentiment {df['score'].mean():+.3f} "
        f"(> +0.15 bullish, < -0.15 bearish)",
        "",
        "Per day:  date | articles | mean_score",
    ]
    for _, r in daily.iterrows():
        lines.append(f"  {r['day']} | {int(r['articles']):>2} | {r['mean_score']:+.3f}")

    lines += ["", f"Headlines (newest first, max {MAX_NEWS_HEADLINES}):"]
    for _, r in df.sort_values("time_published", ascending=False).head(MAX_NEWS_HEADLINES).iterrows():
        lines.append(
            f"  [{r['time_published']:%Y-%m-%d}] ({r['score']:+.2f}) "
            f"{str(r['title'])[:140]} — {r['source']}"
        )
    return "\n".join(lines)


# -------------------------------------------------------------------
# TWITTER TOOL
# -------------------------------------------------------------------

@tool
def get_twitter_posts(ticker: str, start_date: str, end_date: str) -> str:
    """
    Aggregated Twitter/X chatter for `ticker` between start_date and end_date
    (YYYY-MM-DD). The window is capped at the last 14 days before end_date.

    Each tweet is scored -1/0/+1 by a bull/bear keyword lexicon and rolled up
    per day (volume, bull/bear counts, mean polarity, and an
    engagement-weighted mean). A few of the highest-engagement tweets are
    included verbatim for context. Raw tweets are never returned in full.
    """
    ticker = ticker.upper()
    df = _load_window(
        TWITTER_DIR / f"{ticker}_2019.csv", "post_date", start_date, end_date
    )
    if df.empty:
        return f"No tweets found for {ticker} in the requested window."

    for c in ("comment_num", "retweet_num", "like_num"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["engagement"] = df["comment_num"] + df["retweet_num"] + df["like_num"]
    df["polarity"] = df["body"].map(_tweet_polarity)
    df["day"] = df["post_date"].dt.date

    daily = df.groupby("day").agg(
        tweets=("polarity", "size"),
        bull=("polarity", lambda s: int((s > 0).sum())),
        bear=("polarity", lambda s: int((s < 0).sum())),
        net=("polarity", "mean"),
    ).reset_index()

    w = df["engagement"] + 1.0            # +1 so zero-engagement tweets still count
    ew = (df["polarity"] * w).groupby(df["day"]).sum() / w.groupby(df["day"]).sum()
    daily = daily.merge(ew.rename("eng_wtd").reset_index(), on="day", how="left")

    n = len(df)
    lines = [
        f"TWITTER SENTIMENT — {ticker} — {df['day'].min()} to {df['day'].max()}",
        f"{n} tweets | {int((df.polarity > 0).sum())} bullish / "
        f"{int((df.polarity < 0).sum())} bearish / {int((df.polarity == 0).sum())} neutral",
        f"net sentiment {df['polarity'].mean():+.3f} | "
        f"engagement-weighted {(df['polarity'] * w).sum() / w.sum():+.3f}  (range -1..+1)",
        "",
        "Per day:  date | tweets | bull | bear | net | eng_wtd",
    ]
    for _, r in daily.iterrows():
        lines.append(
            f"  {r['day']} | {int(r['tweets']):>5} | {int(r['bull']):>4} | "
            f"{int(r['bear']):>4} | {r['net']:+.2f} | {r['eng_wtd']:+.2f}"
        )

    lines += ["", f"Top tweets by engagement (max {MAX_SAMPLE_TWEETS}):"]
    label = {1: "BULL", -1: "BEAR", 0: "neut"}
    for _, r in df.sort_values("engagement", ascending=False).head(MAX_SAMPLE_TWEETS).iterrows():
        body = " ".join(str(r["body"]).split())[:TWEET_BODY_CHARS]
        lines.append(
            f"  [{r['post_date']:%Y-%m-%d}] {label[r['polarity']]} "
            f"(likes {int(r['like_num'])}, rt {int(r['retweet_num'])}): {body}"
        )
    return "\n".join(lines)


# -------------------------------------------------------------------
# AGENT CREATION
# -------------------------------------------------------------------

def create_sentiment_analyst_agent(model):

    SYSTEM_PROMPT = """
    You are a financial news and social-media sentiment analyst.

    Tools (both take ticker, start_date, end_date as YYYY-MM-DD, and both
    return an already-aggregated summary — you do NOT get raw articles or
    raw tweets):
    - get_news(ticker, start_date, end_date)
        -> per-day article count and mean sentiment score, plus headlines.
    - get_twitter_posts(ticker, start_date, end_date)
        -> per-day tweet volume, bull/bear counts, mean polarity and an
           engagement-weighted mean, plus a few top tweets.

    Request about a 7-day window ending on the analysis date. Windows wider
    than 14 days are clamped automatically.

    How to read the numbers:
    - score / net / eng_wtd are on a -1..+1 scale; near 0 is neutral,
      the engagement-weighted figure tells you where the loud accounts sit.
    - Watch the trend across days and any spike in volume (a catalyst).
    - Compare news vs social — they often diverge.

    Give concise, evidence-backed findings (cite the actual daily numbers).
    """

    tools = [get_news, get_twitter_posts]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    tool_names = ", ".join(tool.name for tool in tools)

    return agent, tool_names
