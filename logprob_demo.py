"""
Log Probability Explorer — Streamlit App
=========================================
An educational tool that visualises token-level log probabilities
returned by OpenAI, Azure OpenAI (Azure AI Foundry), or GitHub Models
chat completions.

Run with:  streamlit run logprob_demo.py
"""

import math
import os
import textwrap
from html import escape

import plotly.graph_objects as go
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────

load_dotenv()

# Determine provider: "azure", "openai", or "github".
# Backward-compat: if PROVIDER is not set, fall back to USE_AZURE.
_provider_env = os.getenv("PROVIDER", "").lower()
if _provider_env in ("azure", "openai", "github"):
    PROVIDER = _provider_env
else:
    # Legacy toggle
    PROVIDER = "azure" if os.getenv("USE_AZURE", "true").lower() in ("true", "1", "yes") else "openai"

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("OPENAI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "") or API_KEY
API_VERSION = os.getenv("API_VERSION", "2025-04-01-preview")

_DEFAULT_MODELS = {"azure": "gpt-5.2", "openai": "gpt-4o", "github": "gpt-4o"}
MODEL = os.getenv("LOGPROB_MODEL", _DEFAULT_MODELS.get(PROVIDER, "gpt-4o"))

GITHUB_MODELS_URL = "https://models.inference.ai.azure.com"

# ──────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────
# Pricing snapshot 2025-Q4. `cached_in` is the discounted price for prompt-cache
# hits when the same prefix is re-used; `None` means the model does not advertise
# a cached price.

PRICING_AS_OF = "2025-Q4"

MODEL_REGISTRY: dict[str, dict] = {
    # ── OpenAI · tokenized via tiktoken ───────────────────────────────
    "gpt-3.5-turbo": {
        "provider": "openai",
        "encoding": "cl100k_base",
        "context_window": 16_385,
        "max_output": 4_096,
        "knowledge_cutoff": "September 2021",
        "input_modalities": ["Text"],
        "output_modalities": ["Text"],
        "features": ["Fine Tuning"],
        "endpoints": ["Chat Completions", "Batch", "Fine Tuning"],
        "pricing": {"input": 0.50, "output": 1.50, "cached_in": None,
                    "batch_in": 0.25, "batch_out": 0.75},
        "performance": "Fast",
        "latency": "Low",
    },
    "gpt-4": {
        "provider": "openai",
        "encoding": "cl100k_base",
        "context_window": 8_192,
        "max_output": 8_192,
        "knowledge_cutoff": "December 2023",
        "input_modalities": ["Text"],
        "output_modalities": ["Text"],
        "features": ["Fine Tuning"],
        "endpoints": ["Chat Completions", "Batch"],
        "pricing": {"input": 30.00, "output": 60.00, "cached_in": None,
                    "batch_in": 15.00, "batch_out": 30.00},
        "performance": "Standard",
        "latency": "Medium",
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "encoding": "cl100k_base",
        "context_window": 128_000,
        "max_output": 4_096,
        "knowledge_cutoff": "December 2023",
        "input_modalities": ["Text", "Image"],
        "output_modalities": ["Text"],
        "features": [],
        "endpoints": ["Chat Completions", "Batch"],
        "pricing": {"input": 10.00, "output": 30.00, "cached_in": None,
                    "batch_in": 5.00, "batch_out": 15.00},
        "performance": "Standard",
        "latency": "Medium",
    },
    "gpt-4o": {
        "provider": "openai",
        "encoding": "o200k_base",
        "context_window": 128_000,
        "max_output": 16_384,
        "knowledge_cutoff": "October 2023",
        "input_modalities": ["Text", "Image", "Audio"],
        "output_modalities": ["Text"],
        "features": ["Fine Tuning", "Structured Outputs"],
        "endpoints": ["Chat Completions", "Responses", "Batch", "Fine Tuning"],
        "pricing": {"input": 2.50, "output": 10.00, "cached_in": 1.25,
                    "batch_in": 1.25, "batch_out": 5.00},
        "performance": "Fast",
        "latency": "Low",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "encoding": "o200k_base",
        "context_window": 128_000,
        "max_output": 16_384,
        "knowledge_cutoff": "October 2023",
        "input_modalities": ["Text", "Image"],
        "output_modalities": ["Text"],
        "features": ["Fine Tuning", "Structured Outputs"],
        "endpoints": ["Chat Completions", "Responses", "Batch", "Fine Tuning"],
        "pricing": {"input": 0.15, "output": 0.60, "cached_in": 0.075,
                    "batch_in": 0.075, "batch_out": 0.30},
        "performance": "Very fast",
        "latency": "Very low",
    },
    "gpt-4.1": {
        "provider": "openai",
        "encoding": "o200k_base",
        "context_window": 1_047_576,
        "max_output": 32_768,
        "knowledge_cutoff": "June 2024",
        "input_modalities": ["Text", "Image"],
        "output_modalities": ["Text"],
        "features": ["Structured Outputs", "Tool Use"],
        "endpoints": ["Chat Completions", "Responses", "Batch"],
        "pricing": {"input": 2.00, "output": 8.00, "cached_in": 0.50,
                    "batch_in": 1.00, "batch_out": 4.00},
        "performance": "Standard",
        "latency": "Medium",
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "encoding": "o200k_base",
        "context_window": 1_047_576,
        "max_output": 32_768,
        "knowledge_cutoff": "June 2024",
        "input_modalities": ["Text", "Image"],
        "output_modalities": ["Text"],
        "features": ["Structured Outputs", "Tool Use"],
        "endpoints": ["Chat Completions", "Responses", "Batch"],
        "pricing": {"input": 0.40, "output": 1.60, "cached_in": 0.10,
                    "batch_in": 0.20, "batch_out": 0.80},
        "performance": "Fast",
        "latency": "Low",
    },
    "gpt-5": {
        "provider": "openai",
        "encoding": "o200k_base",
        "context_window": 400_000,
        "max_output": 128_000,
        "knowledge_cutoff": "September 2024",
        "input_modalities": ["Text", "Image"],
        "output_modalities": ["Text"],
        "features": ["Structured Outputs", "Tool Use", "Reasoning"],
        "endpoints": ["Chat Completions", "Responses", "Batch"],
        "pricing": {"input": 1.25, "output": 10.00, "cached_in": 0.125,
                    "batch_in": 0.625, "batch_out": 5.00},
        "performance": "Reasoning",
        "latency": "High",
    },
}


def available_models() -> list[str]:
    """Return list of models we can tokenize locally via tiktoken."""
    return list(MODEL_REGISTRY.keys())

PYTHON_SAMPLE = textwrap.dedent('''\
    """Customer support triage pipeline.

    Routes inbound messages to the right specialist based on intent,
    sentiment, and the customer\'s subscription tier.
    """

    from dataclasses import dataclass, field
    from datetime import datetime
    from enum import Enum


    class Priority(Enum):
        LOW = "low"
        NORMAL = "normal"
        HIGH = "high"
        URGENT = "urgent"


    @dataclass
    class Ticket:
        id: str
        customer_email: str
        subject: str
        body: str
        created_at: datetime = field(default_factory=datetime.utcnow)
        priority: Priority = Priority.NORMAL
        tags: list[str] = field(default_factory=list)

        def is_premium(self) -> bool:
            return self.customer_email.endswith("@enterprise.example")


    PRIORITY_KEYWORDS = {
        Priority.URGENT: ("outage", "data loss", "security", "can\'t login"),
        Priority.HIGH: ("billing", "refund", "crash", "error 500"),
        Priority.NORMAL: ("question", "how do I", "feature"),
    }


    def classify(ticket: Ticket) -> Priority:
        """Return the inferred priority for a ticket."""
        text = f"{ticket.subject}\\n{ticket.body}".lower()
        for level, keywords in PRIORITY_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return level
        return Priority.LOW if not ticket.is_premium() else Priority.NORMAL


    def summarize(ticket: Ticket, max_chars: int = 140) -> str:
        snippet = ticket.body.strip().replace("\\n", " ")
        if len(snippet) <= max_chars:
            return snippet
        return snippet[: max_chars - 1].rstrip() + "\u2026"


    if __name__ == "__main__":
        sample = Ticket(
            id="T-1042",
            customer_email="alex@enterprise.example",
            subject="Production outage on EU cluster",
            body="Our checkout service has been returning 500s for ~10 minutes.",
        )
        sample.priority = classify(sample)
        print(f"[{sample.priority.value.upper()}] {summarize(sample)}")
''').strip()

MARKDOWN_SAMPLE = textwrap.dedent('''\
    # Acme Search API

    The Acme Search API exposes a small but powerful surface for full-text
    search across **products**, **articles**, and **support threads**.

    > Looking for the v1 reference? See the [legacy docs](https://example.com/v1).

    ## Quick start

    ```bash
    curl -X POST https://api.acme.dev/v2/search \\
      -H "Authorization: Bearer $ACME_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d \'{"query": "wireless headphones", "limit": 5}\'
    ```

    ## Request schema

    | Field   | Type     | Required | Description                                  |
    | ------- | -------- | -------- | -------------------------------------------- |
    | `query` | `string` | yes      | Free-text search query (max **256** chars).  |
    | `limit` | `int`    | no       | Number of hits to return. Defaults to `10`.  |
    | `tags`  | `array`  | no       | Restrict results to one or more tag values.  |

    ## Response example

    ```json
    {
      "hits": [
        {
          "id": "prd_8821",
          "title": "Acme Pulse Wireless Headphones",
          "score": 0.91,
          "tags": ["audio", "wireless"]
        }
      ],
      "took_ms": 23
    }
    ```

    ## Errors

    - `400` \u2013 malformed JSON or missing `query`.
    - `401` \u2013 missing or invalid bearer token.
    - `429` \u2013 rate limit exceeded; retry after `Retry-After` seconds.

    > **Tip:** instrument client retries with exponential backoff and jitter.
''').strip()

TOKENIZER_SAMPLES = {
    "Python (support triage)": PYTHON_SAMPLE,
    "Markdown (API reference)": MARKDOWN_SAMPLE,
}


def _get_openai_client() -> AzureOpenAI | OpenAI:
    """Build an OpenAI / AzureOpenAI / GitHub Models client."""
    if PROVIDER == "azure":
        return AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION,
        )
    if PROVIDER == "github":
        return OpenAI(
            base_url=GITHUB_MODELS_URL,
            api_key=GITHUB_TOKEN,
        )
    return OpenAI(api_key=API_KEY)


def get_provider_config_error() -> str | None:
    """Return a configuration error message for the active chat provider."""
    if PROVIDER == "azure" and (not AZURE_ENDPOINT or not API_KEY):
        return (
            "Missing Azure credentials. Set `AZURE_OPENAI_ENDPOINT` and "
            "`OPENAI_API_KEY` in `.env` to enable model responses."
        )
    if PROVIDER == "github" and not GITHUB_TOKEN:
        return (
            "Missing GitHub token. Set `GITHUB_TOKEN` (or `OPENAI_API_KEY`) "
            "in `.env` to enable model responses."
        )
    if PROVIDER == "openai" and not API_KEY:
        return "Missing OpenAI API key. Set `OPENAI_API_KEY` in `.env` to enable model responses."
    return None

# ──────────────────────────────────────────────
# 2. Page setup
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Log Probability Explorer",
    page_icon="🎲",
    layout="wide",
)

# Custom CSS for token badges
st.markdown(
    """
    <style>
    .token-badge {
        display: inline-block;
        padding: 3px 7px;
        margin: 2px 1px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        font-weight: 600;
        line-height: 1.35;
        white-space: nowrap;
        max-width: 18rem;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .token-id-badge {
        min-width: 4.5rem;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .token-stream-card {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 14px;
    }
    .token-stream-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
    }
    .token-stream-title {
        font-weight: 700;
        font-size: 0.95rem;
    }
    .token-stream-meta {
        font-size: 0.78rem;
        opacity: 0.72;
    }
    .token-stream-body {
        max-height: 260px;
        overflow-y: auto;
        padding-right: 4px;
        line-height: 1.85;
    }
    .token-stream-note {
        margin-top: 8px;
        font-size: 0.78rem;
        opacity: 0.72;
    }
    .prob-high   { background-color: #15803d; color: #f0fdf4; }  /* >90 % */
    .prob-medium { background-color: #a16207; color: #fffbeb; }  /* 50–90 % */
    .prob-low    { background-color: #c2410c; color: #fff7ed; }  /* 10–50 % */
    .prob-vlow   { background-color: #b91c1c; color: #fef2f2; }  /* <10 % */
    .token-control-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin: 4px 0 8px 0;
        padding: 10px 12px;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 10px;
    }
    .token-control-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.78;
    }
    .section-divider {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 2rem 0;
    }
    /* Presenter-style tab pills — scoped to the top-level view switch only
       (keyed by `active_view`) so other st.radio widgets keep their default look. */
    .st-key-active_view div[data-testid="stRadio"] > label { display: none; }
    .st-key-active_view div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        gap: 0.5rem;
        border-bottom: 2px solid rgba(148, 163, 184, 0.25);
        padding: 0;
        margin-bottom: 1.25rem;
        align-items: stretch;
    }
    .st-key-active_view div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 0 0 auto;
        min-width: 160px;
        min-height: 56px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 0 1.5rem;
        border-radius: 10px 10px 0 0;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        background: rgba(148, 163, 184, 0.10);
        transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-bottom: 3px solid transparent;
        margin-bottom: -2px;            /* sit on top of the underline */
        white-space: nowrap;
    }
    .st-key-active_view div[data-testid="stRadio"] div[role="radiogroup"] > label > div {
        margin: 0;                       /* kill default vertical padding */
    }
    .st-key-active_view div[data-testid="stRadio"] div[role="radiogroup"] > label p {
        margin: 0;
        line-height: 1;
    }
    .st-key-active_view div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: rgba(255, 75, 75, 0.12);
    }
    .st-key-active_view div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(180deg, rgba(255, 75, 75, 0.22), rgba(255, 75, 75, 0.04));
        color: #ff6b6b;
        border-color: rgba(255, 75, 75, 0.45);
        border-bottom-color: #ff4b4b;
    }
    .st-key-active_view div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child { display: none; }

    /* Insight + metric cards (right rail / dashboard look) */
    .lp-card {
        background: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .lp-card.lp-inset {
        background: rgba(148, 163, 184, 0.06);
        border-color: rgba(148, 163, 184, 0.20);
        padding: 12px 14px;
        margin: 0;
    }
    .lp-card.lp-insights { padding: 18px 18px 14px 18px; }
    .lp-card h4 {
        margin: 0 0 14px 0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .lp-card .lp-label, .lp-metric .lp-label, .lp-cost-breakdown .lp-label {
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        opacity: 0.85;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .lp-card .lp-value, .lp-metric .lp-value, .lp-cost-breakdown .lp-value {
        font-size: 1.15rem;
        font-weight: 700;
    }
    .lp-card .lp-sub, .lp-metric .lp-sub {
        font-size: 0.78rem;
        opacity: 0.75;
        margin-top: 2px;
    }
    .lp-row {
        margin-bottom: 10px;
        line-height: 1.9;
    }
    .lp-grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 10px;
    }
    .lp-pill {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px 4px 2px 0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(56, 142, 60, 0.28);
        color: #a5d6a7;
        border: 1px solid rgba(102, 187, 106, 0.55);
    }
    .lp-pill.lp-pill-blue   { background: rgba(33, 150, 243, 0.28); color: #90caf9; border-color: rgba(100, 181, 246, 0.55); }
    .lp-pill.lp-pill-amber  { background: rgba(255, 152, 0, 0.28);  color: #ffd180; border-color: rgba(255, 183, 77, 0.55); }
    .lp-pill.lp-pill-purple { background: rgba(156, 39, 176, 0.28); color: #e1bee7; border-color: rgba(206, 147, 216, 0.55); }
    .lp-metric {
        background: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 12px;
        padding: 14px 16px;
    }
    .lp-metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 14px;
    }
    .lp-metric .lp-value-lg {
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .lp-metric .lp-value-money {
        color: #81c784;
    }
    .lp-progress {
        height: 6px;
        width: 100%;
        background: rgba(148, 163, 184, 0.25);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
    }
    .lp-progress > div {
        height: 100%;
        background: linear-gradient(90deg, #42a5f5, #66bb6a);
    }
    .lp-cost-breakdown {
        display: none; /* costs removed from UI */
    }
    /* Context-window hero card */
    .lp-ctx-card {
        background: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .lp-ctx-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }
    .lp-ctx-title {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin-top: 2px;
    }
    .lp-ctx-num {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1;
    }
    .lp-ctx-of {
        font-size: 0.9rem;
        opacity: 0.7;
    }
    .lp-ctx-pct {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
        padding: 8px 14px;
        border-radius: 10px;
        background: rgba(148, 163, 184, 0.15);
    }
    .lp-ctx-pct.lp-ctx-ok     { color: #66bb6a; background: rgba(102, 187, 106, 0.15); }
    .lp-ctx-pct.lp-ctx-warn   { color: #ffb74d; background: rgba(255, 183, 77, 0.18); }
    .lp-ctx-pct.lp-ctx-danger { color: #ef5350; background: rgba(239, 83, 80, 0.20); }
    .lp-ctx-bar {
        position: relative;
        height: 22px;
        width: 100%;
        background: rgba(148, 163, 184, 0.20);
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.30);
    }
    .lp-ctx-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.25s ease;
    }
    .lp-ctx-fill.lp-ctx-ok     { background: linear-gradient(90deg, #2e7d32, #66bb6a); }
    .lp-ctx-fill.lp-ctx-warn   { background: linear-gradient(90deg, #ef6c00, #ffb74d); }
    .lp-ctx-fill.lp-ctx-danger { background: linear-gradient(90deg, #c62828, #ef5350); }
    .lp-ctx-marker {
        position: absolute;
        top: -2px;
        bottom: -2px;
        width: 2px;
        background: #ffffff;
        opacity: 0.85;
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);
    }
    .lp-ctx-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 10px;
        font-size: 0.82rem;
    }
    .lp-ctx-tag {
        font-weight: 600;
        opacity: 0.9;
    }
    .lp-ctx-tag.lp-ctx-ok     { color: #66bb6a; }
    .lp-ctx-tag.lp-ctx-warn   { color: #ffb74d; }
    .lp-ctx-tag.lp-ctx-danger { color: #ef5350; }
    .lp-ctx-tag.lp-ctx-neutral{ opacity: 0.7; }
    .lp-ctx-tag.lp-ctx-marker-tag{ opacity: 0.7; }
    /* Light theme tweaks */
    @media (prefers-color-scheme: light) {
      .lp-card, .lp-metric, .lp-ctx-card {
        background: rgba(15, 23, 42, 0.04);
        border-color: rgba(15, 23, 42, 0.14);
      }
      .lp-card.lp-inset { background: rgba(15, 23, 42, 0.02); }
      .lp-ctx-marker { background: #1f2937; }
      .lp-pill          { color: #2e7d32; background: rgba(56, 142, 60, 0.14); border-color: rgba(56, 142, 60, 0.45); }
      .lp-pill.lp-pill-blue   { color: #1565c0; background: rgba(21, 101, 192, 0.12); border-color: rgba(21, 101, 192, 0.45); }
      .lp-pill.lp-pill-amber  { color: #b26a00; background: rgba(245, 124, 0, 0.14);  border-color: rgba(245, 124, 0, 0.45); }
      .lp-pill.lp-pill-purple { color: #6a1b9a; background: rgba(106, 27, 154, 0.12); border-color: rgba(106, 27, 154, 0.45); }
      .lp-metric .lp-value-money { color: #2e7d32; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 3. Title & active view selection (presenter tabs)
# ──────────────────────────────────────────────

st.title("🎲  Log Probability Explorer")
if PROVIDER == "azure":
    st.caption(
        f"Model: **{MODEL}**  •  Endpoint: `{AZURE_ENDPOINT}`  •  "
        f"API version: `{API_VERSION}`"
    )
elif PROVIDER == "github":
    st.caption(
        f"Model: **{MODEL}**  •  Provider: GitHub Models  •  "
        f"Endpoint: `{GITHUB_MODELS_URL}`"
    )
else:
    st.caption(f"Model: **{MODEL}**  •  Provider: OpenAI API")

active_view = st.radio(
    "View",
    options=["🧪  Tokenizer", "🎲  Logprob"],
    horizontal=True,
    key="active_view",
    label_visibility="collapsed",
)

# ──────────────────────────────────────────────
# 4. Sidebar — context-aware explanation & controls
# ──────────────────────────────────────────────

with st.sidebar:
    if "Tokenizer" in active_view:
        st.header("🧪  Tokenizer")
        st.markdown(
            textwrap.dedent("""\
            Models read **tokens**, not characters. Code, punctuation,
            whitespace and non-Latin text can fill the context faster than
            plain English.
            """)
        )

        st.divider()
        st.header("🔍  Reading The View")
        st.markdown(
            textwrap.dedent("""\
            - The big bar is **input tokens / context window**.
            - The marker reserves room for max output.
            - Token chips are a compact preview; hover to inspect exact text.
            """)
        )

        st.divider()
        st.header("📐  Headroom")
        st.markdown(
            textwrap.dedent("""\
            **Free in context** is total remaining space.
            **Headroom for output** is what's left after reserving the
            model's max-output budget.
            """)
        )

        top_logprobs_n = 5
        max_tokens = 200
        temperature = 0.7
    else:
        st.header("🎲  About Log Probabilities")
        st.markdown(
            textwrap.dedent("""\
            At every step the model assigns a **probability** to each
            possible next token. The natural log of that probability is
            the **logprob**:

            $$p = e^{\\text{logprob}}$$

            | logprob | probability |
            |--------:|------------:|
            |    0.00 |      100 %  |
            |   −0.10 |       90 %  |
            |   −0.69 |       50 %  |
            |   −2.30 |       10 %  |
            |   −4.61 |        1 %  |

            Closer to 0 → very confident. Very negative → many plausible
            alternatives.
            """)
        )

        st.divider()
        st.header("⚙️  Generation Settings")
        top_logprobs_n = st.slider(
            "Top-N alternatives per token",
            min_value=1, max_value=5, value=5,
            help="How many alternative tokens to show at each position.",
        )
        max_tokens = st.slider(
            "Max response tokens",
            min_value=50, max_value=500, value=200, step=50,
            help="Upper limit on the number of tokens the model may generate.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=2.0, value=0.7, step=0.1,
            help="Controls randomness. 0 = deterministic, 2 = very creative. "
                 "Higher temperature → more varied output → lower per-token probabilities.",
        )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

PASTEL_COLORS = [
    "#2563eb", "#15803d", "#a16207", "#be185d",
    "#7c3aed", "#0f766e", "#c2410c", "#0369a1",
    "#4d7c0f", "#854d0e", "#b91c1c", "#6d28d9",
]


def get_encoder(model_name: str = MODEL) -> tiktoken.Encoding:
    """Return a tiktoken encoder for a model, falling back to o200k_base."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def tokenise_text(text: str, model_name: str = MODEL) -> list[tuple[int, str]]:
    """Encode *text* into (token_id, token_string) pairs for a specific model."""
    enc = get_encoder(model_name)
    ids = enc.encode(text)
    return [(tid, enc.decode([tid])) for tid in ids]


def prob_css_class(probability: float) -> str:
    """Return a CSS class name based on probability percentage."""
    if probability > 90:
        return "prob-high"
    if probability > 50:
        return "prob-medium"
    if probability > 10:
        return "prob-low"
    return "prob-vlow"


def render_token_badges(
    tokens: list[str],
    colors: list[str] | None = None,
    probabilities: list[float] | None = None,
) -> str:
    """Build an HTML string of styled inline badges for a list of tokens."""
    html_parts: list[str] = []
    for i, tok in enumerate(tokens):
        # Show whitespace explicitly, then escape once for text content.
        display_raw = tok.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥")
        if len(display_raw) > 48:
            display_raw = f"{display_raw[:45]}..."
        display = escape(display_raw, quote=False)
        if probabilities is not None:
            css = prob_css_class(probabilities[i])
            html_parts.append(
                f'<span class="token-badge {css}" data-token-index="{i}">{display}</span>'
            )
        else:
            bg = (colors or PASTEL_COLORS)[i % len(colors or PASTEL_COLORS)]
            html_parts.append(
                f'<span class="token-badge" '
                f'style="background:{bg};" data-token-index="{i}">{display}</span>'
            )
    return "".join(html_parts)


def render_token_id_badges(token_ids: list[int]) -> str:
    """Build an HTML string of styled badges for token IDs only."""
    html_parts: list[str] = []
    for index, token_id in enumerate(token_ids):
        bg = PASTEL_COLORS[index % len(PASTEL_COLORS)]
        html_parts.append(
            f'<span class="token-badge token-id-badge" '
            f'style="background:{bg};" data-token-index="{index}">{token_id}</span>'
        )
    return "".join(html_parts)


def logprob_to_pct(lp: float) -> float:
    return math.exp(lp) * 100


def _format_int(n: int) -> str:
    """Compact human-friendly token count (1.2K / 3.4M)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return f"{n:,}"


def render_token_view(
    text: str,
    model_name: str,
    show_all: bool,
    token_display: str,
) -> None:
    """Render the left-hand tokenization view focused on context-window usage."""
    spec = MODEL_REGISTRY[model_name]

    if not text.strip():
        st.warning("Type or paste some text to see how it tokenizes.")
        return

    token_pairs = tokenise_text(text, model_name)
    token_count = len(token_pairs)
    encoding_label = get_encoder(model_name).name
    source_label = f"`tiktoken` · {encoding_label}"

    context_window = spec["context_window"]
    max_output = spec["max_output"]
    used_pct = (token_count / context_window) * 100 if context_window else 0
    remaining = max(context_window - token_count, 0)
    # Headroom = how much fits if we also reserve room for max_output tokens.
    reservable = max(context_window - max_output, 0)
    headroom = max(reservable - token_count, 0)
    over_budget = token_count > reservable

    char_count = len(text)
    tokens_per_100 = (token_count / char_count * 100) if char_count else 0

    if used_pct < 50:
        bar_class = "ok"
        status_label = "Plenty of headroom"
    elif used_pct < 80:
        bar_class = "warn"
        status_label = "Filling up"
    else:
        bar_class = "danger"
        status_label = "Near limit"

    # Where does the "max output reservation" sit on the bar?
    output_marker_pct = (
        (reservable / context_window) * 100 if context_window else 100
    )

    # ── Hero: context-window gauge ───────────────────────────────
    st.markdown(
        f"""
        <div class="lp-ctx-card">
          <div class="lp-ctx-head">
            <div>
              <div class="lp-label">Context window usage</div>
              <div class="lp-ctx-title">
                <span class="lp-ctx-num">{token_count:,}</span>
                <span class="lp-ctx-of">/ {context_window:,} tokens</span>
              </div>
              <div class="lp-sub">{status_label} · {used_pct:.2f}% full</div>
            </div>
            <div class="lp-ctx-pct lp-ctx-{bar_class}">{used_pct:.1f}%</div>
          </div>
          <div class="lp-ctx-bar">
            <div class="lp-ctx-fill lp-ctx-{bar_class}" style="width:{min(used_pct, 100):.3f}%"></div>
            <div class="lp-ctx-marker" style="left:{min(output_marker_pct, 100):.3f}%"
                 title="Reserved for up to {max_output:,} output tokens"></div>
          </div>
          <div class="lp-ctx-legend">
            <span class="lp-ctx-tag lp-ctx-{bar_class}">● Used · {_format_int(token_count)}</span>
            <span class="lp-ctx-tag lp-ctx-neutral">○ Free · {_format_int(remaining)}</span>
            <span class="lp-ctx-tag lp-ctx-marker-tag">▌ Output reservation · {_format_int(max_output)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if over_budget:
        st.error(
            f"Your prompt ({token_count:,} tokens) leaves less than the model's "
            f"max-output budget ({max_output:,} tokens) of headroom. Trim the "
            "prompt or the model will refuse / truncate the response."
        )

    # ── Secondary stats ──────────────────────────────────────────
    st.markdown(
        f"""
        <div class="lp-metric-row">
          <div class="lp-metric">
            <div class="lp-label">Prompt tokens</div>
            <div class="lp-value-lg">{token_count:,}</div>
            <div class="lp-sub">{tokens_per_100:.2f} tokens / 100 chars · {char_count:,} chars</div>
          </div>
          <div class="lp-metric">
            <div class="lp-label">Headroom for output</div>
            <div class="lp-value-lg">{_format_int(headroom)}</div>
            <div class="lp-sub">After reserving {max_output:,} max-output tokens</div>
          </div>
          <div class="lp-metric">
            <div class="lp-label">Free in context</div>
            <div class="lp-value-lg">{_format_int(remaining)}</div>
            <div class="lp-sub">{(remaining / context_window * 100) if context_window else 0:.1f}% of window</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Token pieces ─────────────────────────────────────────────
    st.markdown('<div class="token-control-label">Token stream controls</div>', unsafe_allow_html=True)
    control_col1, control_col2 = st.columns([1, 1.2], vertical_alignment="center")
    show_all = control_col1.toggle(
        "Full token stream",
        value=show_all,
        help="Render every token inside the token stream box. Keep this off for long prompts.",
        key="tokenizer_show_all_inline",
    )
    token_display = control_col2.radio(
        "Token display",
        options=["Text", "Token IDs"],
        horizontal=True,
        key="tokenizer_token_display_inline",
        index=0 if token_display == "Text" else 1,
        label_visibility="collapsed",
    )

    preview_limit = 160
    display_pairs = token_pairs if show_all else token_pairs[:preview_limit]
    hidden_count = max(len(token_pairs) - len(display_pairs), 0)
    token_body = (
        render_token_id_badges([token_id for token_id, _ in display_pairs])
        if token_display == "Token IDs" else
        render_token_badges([token for _, token in display_pairs])
    )
    token_note = (
        f"Showing {len(display_pairs):,} of {len(token_pairs):,} tokens. "
        f"{hidden_count:,} hidden to keep the view readable."
        if hidden_count else
        f"Showing all {len(display_pairs):,} tokens."
    )
    st.markdown(
        f"""
        <div class="token-stream-card">
          <div class="token-stream-head">
            <div class="token-stream-title">Token stream</div>
            <div class="token-stream-meta">{source_label}</div>
          </div>
          <div class="token-stream-body">
            {token_body}
          </div>
          <div class="token-stream-note">{token_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if token_pairs:
        with st.expander("Token table", expanded=False):
            st.dataframe(
                [
                    {"#": i + 1, "Token ID": tid, "Token": repr(tok), "Characters": len(tok)}
                    for i, (tid, tok) in enumerate(token_pairs)
                ],
                width='stretch',
                hide_index=True,
            )


def _pill(text: str, variant: str = "") -> str:
    cls = f"lp-pill {variant}" if variant else "lp-pill"
    return f'<span class="{cls}">{text}</span>'


def render_model_insights(model_name: str) -> None:
    """Render the right-hand 'Model insights' card."""
    spec = MODEL_REGISTRY[model_name]

    perf_emoji = {"Very fast": "⚡⚡", "Fast": "⚡", "Standard": "⚙️", "Reasoning": "🧠"}
    latency_emoji = {"Very low": "🟢", "Low": "🕐", "Medium": "🕑", "High": "🕒"}

    in_modalities_html = "".join(_pill(m, "lp-pill-blue") for m in spec["input_modalities"])
    out_modalities_html = "".join(_pill(m, "lp-pill-blue") for m in spec["output_modalities"])
    features_html = (
        "".join(_pill(f, "lp-pill-purple") for f in spec["features"]) or
        '<span class="lp-sub">—</span>'
    )
    endpoints_html = "".join(_pill(e, "lp-pill-amber") for e in spec["endpoints"])

    html = (
        '<div class="lp-card lp-insights">'
        '<h4>Model insights</h4>'
        '<div class="lp-card lp-inset"><div class="lp-label">Knowledge cutoff</div>'
        f'<div class="lp-value">{spec["knowledge_cutoff"]}</div></div>'
        '<div class="lp-grid-2">'
        '<div class="lp-card lp-inset"><div class="lp-label">Context window</div>'
        f'<div class="lp-value">{spec["context_window"]:,} <span class="lp-sub">tok</span></div></div>'
        '<div class="lp-card lp-inset"><div class="lp-label">Max output</div>'
        f'<div class="lp-value">{spec["max_output"]:,} <span class="lp-sub">tok</span></div></div>'
        '</div>'
        '<div class="lp-grid-2">'
        '<div class="lp-card lp-inset"><div class="lp-label">Performance</div>'
        f'<div class="lp-value">{perf_emoji.get(spec["performance"], "")} '
        f'<span class="lp-sub">{spec["performance"]}</span></div></div>'
        '<div class="lp-card lp-inset"><div class="lp-label">Latency</div>'
        f'<div class="lp-value">{latency_emoji.get(spec["latency"], "")} '
        f'<span class="lp-sub">{spec["latency"]}</span></div></div>'
        '</div>'
        '<div class="lp-label">Input modalities</div>'
        f'<div class="lp-row">{in_modalities_html}</div>'
        '<div class="lp-label">Output modalities</div>'
        f'<div class="lp-row">{out_modalities_html}</div>'
        '<div class="lp-label">Supported features</div>'
        f'<div class="lp-row">{features_html}</div>'
        '<div class="lp-label">Supported endpoints</div>'
        f'<div class="lp-row">{endpoints_html}</div>'
        '<div class="lp-sub" style="margin-top:6px;">Tokenization via '
        '<code>tiktoken</code>.</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_logprob_explorer(provider_config_error: str | None) -> None:
    """Render the log probability explorer inside its own tab."""
    if provider_config_error:
        st.info(provider_config_error)

    prompt = st.text_area(
        "Enter your prompt",
        value="Explain in one sentence why the sky is blue.",
        height=100,
    )

    submit = st.button(
        "🚀  Send to model",
        type="primary",
        width='stretch',
        disabled=bool(provider_config_error),
    )

    if submit and prompt.strip():
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("📝  Prompt Tokenisation")

        token_pairs = tokenise_text(prompt)
        token_strs = [t for _, t in token_pairs]

        st.markdown(
            render_token_badges(token_strs),
            unsafe_allow_html=True,
        )
        st.caption(f"**{len(token_pairs)}** tokens  •  encoding: `{get_encoder().name}`")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("🤖  Model Response")

        try:
            client = _get_openai_client()

            with st.spinner("Waiting for model response …"):
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    logprobs=True,
                    top_logprobs=top_logprobs_n,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )

        except Exception as exc:
            st.error(f"**API error:** {exc}")
            st.stop()

        choice = response.choices[0]
        reply_text = choice.message.content or ""
        logprobs_data = choice.logprobs

        st.markdown(f"> {reply_text}")

        if response.usage:
            usage = response.usage
            col1, col2, col3 = st.columns(3)
            col1.metric("Prompt tokens", usage.prompt_tokens)
            col2.metric("Completion tokens", usage.completion_tokens)
            col3.metric("Total tokens", usage.total_tokens)

        if logprobs_data and logprobs_data.content:
            tokens_info = logprobs_data.content
            table_rows = []
            chart_tokens = []
            chart_probs = []

            for idx, token_info in enumerate(tokens_info):
                pct = logprob_to_pct(token_info.logprob)
                alts = ", ".join(
                    f"{alt.token!r} ({logprob_to_pct(alt.logprob):.1f}%)"
                    for alt in (token_info.top_logprobs or [])
                    if alt.token != token_info.token
                )
                table_rows.append(
                    {
                        "#": idx + 1,
                        "Token": repr(token_info.token),
                        "Log Prob": f"{token_info.logprob:.4f}",
                        "Probability %": f"{pct:.1f}",
                        "Top Alternatives": alts or "—",
                    }
                )
                chart_tokens.append(token_info.token)
                chart_probs.append(pct)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.subheader("🎨  Response Tokens — Colour-Coded by Confidence")
            st.markdown(
                render_token_badges(
                    chart_tokens,
                    probabilities=chart_probs,
                ),
                unsafe_allow_html=True,
            )
            st.caption("🟩 >90 %  · 🟨 50–90 %  · 🟧 10–50 %  · 🟥 <10 %")

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.subheader("📊  Token-by-Token Log Probabilities")
            st.dataframe(
                table_rows,
                width='stretch',
                hide_index=True,
            )

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.subheader("📈  Probability Overview Chart")

            bar_colors = [
                "#15803d" if p > 90 else
                "#a16207" if p > 50 else
                "#c2410c" if p > 10 else
                "#b91c1c"
                for p in chart_probs
            ]

            fig = go.Figure(
                go.Bar(
                    x=list(range(1, len(chart_tokens) + 1)),
                    y=chart_probs,
                    text=[repr(token) for token in chart_tokens],
                    textposition="outside",
                    marker_color=bar_colors,
                    hovertemplate=(
                        "<b>Token %{x}</b>: %{text}<br>"
                        "Probability: %{y:.1f} %<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                xaxis_title="Token position",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 105],
                height=420,
                margin=dict(t=30, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(148,163,184,0.06)",
                font=dict(color="#e5e7eb"),
            )
            fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.25)")
            fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.25)")
            st.plotly_chart(fig, width='stretch')

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.subheader("🔍  Token Details — Explore Alternatives")
            st.caption("Click any token to see what else the model considered.")

            for idx, token_info in enumerate(tokens_info):
                pct = logprob_to_pct(token_info.logprob)
                label = f"Token {idx + 1}:  {token_info.token!r}  —  {pct:.1f} %"
                with st.expander(label):
                    if token_info.top_logprobs:
                        alt_tokens = [alt.token for alt in token_info.top_logprobs]
                        alt_probs = [logprob_to_pct(alt.logprob) for alt in token_info.top_logprobs]

                        fig_alt = go.Figure(
                            go.Bar(
                                y=[repr(token) for token in alt_tokens],
                                x=alt_probs,
                                orientation="h",
                                marker_color=[
                                    "#15803d" if token == token_info.token else "#64748b"
                                    for token in alt_tokens
                                ],
                                hovertemplate=(
                                    "<b>%{y}</b><br>"
                                    "Probability: %{x:.1f} %<extra></extra>"
                                ),
                            )
                        )
                        fig_alt.update_layout(
                            xaxis_title="Probability (%)",
                            xaxis_range=[0, 105],
                            height=40 + 32 * len(alt_tokens),
                            margin=dict(l=10, r=10, t=10, b=30),
                            yaxis=dict(autorange="reversed"),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(148,163,184,0.06)",
                            font=dict(color="#e5e7eb"),
                        )
                        fig_alt.update_xaxes(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.25)")
                        fig_alt.update_yaxes(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.25)")
                        st.plotly_chart(fig_alt, width='stretch')
                    else:
                        st.info("No alternative tokens returned for this position.")
        else:
            st.warning("The model did not return log probability data.")

    elif submit:
        st.warning("Please enter a prompt first.")


provider_config_error = get_provider_config_error()
if "tokenizer_playground_applied_preset" not in st.session_state:
    default_preset = next(iter(TOKENIZER_SAMPLES))
    st.session_state["tokenizer_playground_applied_preset"] = default_preset
    st.session_state["tokenizer_playground_preset"] = default_preset
    st.session_state["tokenizer_playground_text"] = TOKENIZER_SAMPLES[default_preset]

if "Tokenizer" in active_view:
    st.subheader("🧪  Tokenizer Playground")
    st.caption(
        "Pick a model, edit the sample, and watch how much of the model's "
        "**context window** your prompt actually consumes."
    )

    models = available_models()
    if not models:
        st.error("No tokenizable models available.")
        st.stop()

    selected_model = st.selectbox(
        "Model",
        options=models,
        format_func=lambda m: f"OpenAI · {m}",
        key="tokenizer_selected_model",
    )

    spec = MODEL_REGISTRY[selected_model]

    preset = st.selectbox(
        "Sample",
        options=list(TOKENIZER_SAMPLES),
        key="tokenizer_playground_preset",
    )
    if st.session_state["tokenizer_playground_applied_preset"] != preset:
        st.session_state["tokenizer_playground_text"] = TOKENIZER_SAMPLES[preset]
        st.session_state["tokenizer_playground_applied_preset"] = preset

    left_col, right_col = st.columns([2.4, 1])

    with left_col:
        # If the widget state was cleared (e.g. on first paint), fall back to
        # the currently-selected preset so the sample is always visible.
        if not st.session_state.get("tokenizer_playground_text"):
            st.session_state["tokenizer_playground_text"] = TOKENIZER_SAMPLES[preset]

        playground_text = st.text_area(
            "Tokenizer input",
            key="tokenizer_playground_text",
            height=300,
            help="Edit the sample to see how the selected model splits it.",
        )
        render_token_view(
            playground_text,
            model_name=selected_model,
            show_all=st.session_state.get("tokenizer_show_all_inline", False),
            token_display=st.session_state.get("tokenizer_token_display_inline", "Text"),
        )

    with right_col:
        render_model_insights(selected_model)
else:
    render_logprob_explorer(provider_config_error)


# ──────────────────────────────────────────────
# 5. Main logic — runs after the user clicks Send
# ──────────────────────────────────────────────
