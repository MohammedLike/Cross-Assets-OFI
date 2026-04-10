"""
Lightweight RAG (Retrieval-Augmented Generation) news pipeline for OFI study.

Goal: Combine market microstructure (OFI) with information shocks (news).

Pipeline:
  1. Fetch financial news headlines (RSS feeds — free, no API key)
  2. Embed using sentence-transformers
  3. Index in FAISS for fast similarity search
  4. At each timestep t, retrieve relevant news within a lookback window
  5. Convert to sentiment score / aggregate embedding
  6. Add as features alongside OFI

Why RAG (not full LLM): we don't need text generation — we need
information retrieval as a feature engineering step. This keeps the
system fast, deterministic, and interpretable.
"""
from __future__ import annotations
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Optional dependencies — degrade gracefully if missing
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ── RSS feeds for Indian financial news ───────────────────────────────

INDIA_FINANCE_FEEDS = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "businessstandard_markets": "https://www.business-standard.com/rss/markets-106.rss",
}


# ── Simple sentiment lexicon (fallback when no model available) ──────

POSITIVE_WORDS = {
    "rise", "rises", "rising", "rose", "rally", "rallies", "rallied", "gain",
    "gains", "gained", "surge", "surges", "surged", "jump", "jumps", "jumped",
    "climb", "climbs", "climbed", "soar", "soars", "soared", "advance",
    "advances", "advanced", "high", "highs", "record", "boost", "boosts",
    "boosted", "upbeat", "bullish", "outperform", "beat", "beats", "growth",
    "grew", "strong", "stronger", "strongest", "positive", "profit", "profits",
    "earnings beat", "upgrade", "upgrades", "upgraded", "buy",
}

NEGATIVE_WORDS = {
    "fall", "falls", "falling", "fell", "drop", "drops", "dropped", "decline",
    "declines", "declined", "plunge", "plunges", "plunged", "tumble", "tumbles",
    "tumbled", "slump", "slumps", "slumped", "slide", "slides", "slid", "lose",
    "loses", "lost", "loss", "losses", "weak", "weaker", "weakest", "low",
    "lows", "bearish", "downbeat", "underperform", "miss", "misses", "missed",
    "concern", "concerns", "concerned", "warning", "warns", "warned", "fear",
    "fears", "feared", "downgrade", "downgrades", "downgraded", "sell",
    "negative", "crisis", "crash", "crashed",
}


# ── News fetching ─────────────────────────────────────────────────────

def fetch_rss_feed(url: str, max_items: int = 100) -> list[dict]:
    """Fetch one RSS feed and return list of {title, summary, published, link}."""
    if not HAS_FEEDPARSER:
        return []
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            items.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:500],
                "published": published.isoformat() if published else None,
                "link": entry.get("link", ""),
            })
        return items
    except Exception as e:
        warnings.warn(f"Failed to fetch {url}: {e}")
        return []


def fetch_all_news(feeds: dict = None, max_per_feed: int = 100) -> pd.DataFrame:
    """
    Fetch all configured RSS feeds and return as DataFrame.

    Returns DataFrame with columns: source, title, summary, published, link.
    """
    feeds = feeds or INDIA_FINANCE_FEEDS
    rows = []
    for source, url in feeds.items():
        items = fetch_rss_feed(url, max_per_feed)
        for item in items:
            item["source"] = source
            rows.append(item)
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        df = df.dropna(subset=["title"]).drop_duplicates(subset=["title"])
        df = df.sort_values("published", ascending=False).reset_index(drop=True)
    return df


# ── Sentiment scoring ─────────────────────────────────────────────────

def lexicon_sentiment(text: str) -> float:
    """
    Simple bag-of-words sentiment score in [-1, +1].
    Used as fallback when no embedding model is available.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    words = text.lower().split()
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def add_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment score column to news DataFrame."""
    if len(news_df) == 0:
        return news_df
    news_df = news_df.copy()
    text = news_df["title"].fillna("") + " " + news_df["summary"].fillna("")
    news_df["sentiment"] = text.apply(lexicon_sentiment)
    return news_df


# ── RAG Index ─────────────────────────────────────────────────────────

class NewsRAG:
    """
    Retrieval-Augmented index over news headlines.

    Embeds news with sentence-transformers, indexes with FAISS,
    and supports time-filtered similarity retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.news_df = None
        self.embeddings = None

        if HAS_ST:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                warnings.warn(f"Could not load sentence-transformer: {e}")

    def build(self, news_df: pd.DataFrame):
        """Embed and index news."""
        if len(news_df) == 0:
            self.news_df = news_df
            return self

        self.news_df = news_df.reset_index(drop=True)
        texts = (news_df["title"].fillna("") + " " + news_df["summary"].fillna("")).tolist()

        if self.model is None:
            # Fallback: random embeddings (still allows the pipeline to run)
            self.embeddings = np.random.RandomState(42).randn(len(texts), 32).astype(np.float32)
        else:
            self.embeddings = self.model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            ).astype(np.float32)

        if HAS_FAISS and self.embeddings is not None:
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings)

        return self

    def retrieve(
        self,
        query: str,
        timestamp: Optional[pd.Timestamp] = None,
        lookback_hours: int = 24,
        k: int = 5,
    ) -> pd.DataFrame:
        """
        Retrieve top-k news articles similar to query, filtered by time.

        Parameters
        ----------
        query : str — query text (e.g. "BankNifty rate decision")
        timestamp : optional — only include news published BEFORE this time
        lookback_hours : int — max age of returned news
        k : int — number of results

        Returns
        -------
        DataFrame slice of news_df with similarity scores.
        """
        if self.news_df is None or len(self.news_df) == 0:
            return pd.DataFrame()

        if self.model is None or self.index is None:
            # No embedding model: fall back to keyword filter
            mask = self.news_df["title"].str.contains(query, case=False, na=False)
            return self.news_df[mask].head(k)

        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(q_emb, min(k * 5, len(self.news_df)))
        results = self.news_df.iloc[indices[0]].copy()
        results["distance"] = distances[0]

        # Time filter
        if timestamp is not None and "published" in results.columns:
            tmin = timestamp - pd.Timedelta(hours=lookback_hours)
            results = results[
                (results["published"] >= tmin) & (results["published"] <= timestamp)
            ]

        return results.head(k)


# ── Time-aligned news features ────────────────────────────────────────

def build_news_features(
    news_df: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    window_hours: int = 4,
    asset_keywords: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build per-bar news features aligned to OFI bar index.

    For each bar timestamp t, aggregates news from [t - window_hours, t]:
      - news_count       : number of articles in window
      - news_sentiment   : mean sentiment of those articles
      - asset_mentions_X : count of mentions of asset X in titles

    Parameters
    ----------
    news_df : output of fetch_all_news + add_sentiment
    bar_index : DatetimeIndex of OFI bars
    window_hours : lookback window
    asset_keywords : dict mapping asset name to list of keywords to count
                     e.g. {'NIFTY': ['nifty', 'index'], 'BANKNIFTY': ['bank nifty', 'banknifty']}

    Returns
    -------
    DataFrame indexed by bar_index with news feature columns.
    """
    if asset_keywords is None:
        asset_keywords = {
            "NIFTY":     ["nifty", "nifty 50", "index"],
            "BANKNIFTY": ["bank nifty", "banknifty", "banking"],
            "HDFCBANK":  ["hdfc", "hdfc bank"],
            "RELIANCE":  ["reliance", "ril"],
            "INFY":      ["infosys", "infy", "it sector"],
        }

    features = pd.DataFrame(
        {
            "news_count": 0,
            "news_sentiment": 0.0,
            **{f"mentions_{k}": 0 for k in asset_keywords.keys()},
        },
        index=bar_index,
    )

    if news_df is None or len(news_df) == 0 or "published" not in news_df.columns:
        return features

    nf = news_df.dropna(subset=["published"]).copy()
    if "sentiment" not in nf.columns:
        nf = add_sentiment(nf)
    nf["text"] = (nf["title"].fillna("") + " " + nf["summary"].fillna("")).str.lower()

    window = pd.Timedelta(hours=window_hours)

    # Vectorized: for each bar, get news within window. We sort + use searchsorted.
    nf = nf.sort_values("published").reset_index(drop=True)
    pub_times = nf["published"].values

    for t in bar_index:
        t_np = np.datetime64(t)
        lo = np.searchsorted(pub_times, t_np - np.timedelta64(window))
        hi = np.searchsorted(pub_times, t_np, side="right")
        window_news = nf.iloc[lo:hi]
        if len(window_news) == 0:
            continue
        features.at[t, "news_count"] = len(window_news)
        features.at[t, "news_sentiment"] = window_news["sentiment"].mean()
        for asset, kws in asset_keywords.items():
            cnt = 0
            for kw in kws:
                cnt += window_news["text"].str.contains(kw, na=False).sum()
            features.at[t, f"mentions_{asset}"] = cnt

    return features


# ── End-to-end ────────────────────────────────────────────────────────

def run_news_pipeline(
    bar_index: pd.DatetimeIndex,
    window_hours: int = 4,
    cache_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full RAG news pipeline:
      1. Fetch RSS news
      2. Add sentiment
      3. Align to bar index → news features

    Returns
    -------
    raw_news_df, news_features_df
    """
    if cache_path is not None and cache_path.exists():
        try:
            news_df = pd.read_parquet(cache_path)
        except Exception:
            news_df = fetch_all_news()
    else:
        news_df = fetch_all_news()
        if cache_path is not None and len(news_df) > 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                news_df.to_parquet(cache_path)
            except Exception:
                pass

    news_df = add_sentiment(news_df)
    feats = build_news_features(news_df, bar_index, window_hours=window_hours)
    return news_df, feats
