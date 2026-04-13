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
        padding: 4px 8px;
        margin: 2px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        font-weight: 600;
        line-height: 1.6;
        white-space: pre;
    }
    .prob-high   { background-color: #c8e6c9; color: #1b5e20; }  /* >90 % */
    .prob-medium { background-color: #fff9c4; color: #f57f17; }  /* 50–90 % */
    .prob-low    { background-color: #ffe0b2; color: #e65100; }  /* 10–50 % */
    .prob-vlow   { background-color: #ffcdd2; color: #b71c1c; }  /* <10 % */
    .section-divider {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 3. Sidebar — educational primer & controls
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️  What Are Log Probabilities?")
    st.markdown(
        textwrap.dedent("""\
        Large Language Models (LLMs) don't just produce text — they assign a
        **probability** to every possible next token at each step.

        A **log probability** (logprob) is the natural logarithm of that
        probability:

        $$p = e^{\\text{logprob}}$$

        | logprob | probability |
        |--------:|------------:|
        |    0.00 |      100 %  |
        |   −0.10 |       90 %  |
        |   −0.69 |       50 %  |
        |   −2.30 |       10 %  |
        |   −4.61 |        1 %  |

        A value **closer to 0** means the model was very confident about that
        token.  A very negative value means the model considered many
        alternatives.
        """)
    )

    st.divider()
    st.header("🔧  What Is Tokenisation?")
    st.markdown(
        textwrap.dedent("""\
        Before the model reads your prompt it splits the text into **tokens**
        — small pieces that can be whole words, sub-words, or even single
        characters.

        Understanding tokenisation helps you see *exactly* what the model
        receives and how it generates each piece of its reply.
        """)
    )

    st.divider()
    st.header("⚙️  Settings")
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
# 4. Title & prompt input
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

if PROVIDER == "azure" and (not AZURE_ENDPOINT or not API_KEY):
    st.error(
        "Missing Azure credentials.  Make sure `AZURE_OPENAI_ENDPOINT` and "
        "`OPENAI_API_KEY` are set in your `.env` file."
    )
    st.stop()
elif PROVIDER == "github" and not GITHUB_TOKEN:
    st.error(
        "Missing GitHub token.  Make sure `GITHUB_TOKEN` (or `OPENAI_API_KEY`) "
        "is set in your `.env` file."
    )
    st.stop()
elif PROVIDER == "openai" and not API_KEY:
    st.error(
        "Missing OpenAI API key.  Make sure `OPENAI_API_KEY` is set in your `.env` file."
    )
    st.stop()

prompt = st.text_area(
    "Enter your prompt",
    value="Explain in one sentence why the sky is blue.",
    height=100,
)

submit = st.button("🚀  Send to model", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

PASTEL_COLORS = [
    "#bbdefb", "#c8e6c9", "#fff9c4", "#f8bbd0",
    "#d1c4e9", "#b2dfdb", "#ffe0b2", "#b3e5fc",
    "#dcedc8", "#f0f4c3", "#ffccbc", "#e1bee7",
]


def get_encoder() -> tiktoken.Encoding:
    """Return a tiktoken encoder, falling back to o200k_base."""
    try:
        return tiktoken.encoding_for_model(MODEL)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def tokenise_prompt(text: str) -> list[tuple[int, str]]:
    """Encode *text* into (token_id, token_string) pairs."""
    enc = get_encoder()
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
        display = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Show whitespace explicitly
        display = display.replace(" ", "␣").replace("\n", "↵\n").replace("\t", "⇥")
        if probabilities is not None:
            css = prob_css_class(probabilities[i])
            html_parts.append(
                f'<span class="token-badge {css}" '
                f'title="prob: {probabilities[i]:.1f}%">{display}</span>'
            )
        else:
            bg = (colors or PASTEL_COLORS)[i % len(colors or PASTEL_COLORS)]
            html_parts.append(
                f'<span class="token-badge" '
                f'style="background:{bg};">{display}</span>'
            )
    return "".join(html_parts)


def logprob_to_pct(lp: float) -> float:
    return math.exp(lp) * 100


# ──────────────────────────────────────────────
# 5. Main logic — runs after the user clicks Send
# ──────────────────────────────────────────────


if submit and prompt.strip():
    # ── 5a. Prompt tokenisation ──────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("📝  Prompt Tokenisation")

    token_pairs = tokenise_prompt(prompt)
    token_strs = [t for _, t in token_pairs]

    st.markdown(
        render_token_badges(token_strs),
        unsafe_allow_html=True,
    )
    st.caption(f"**{len(token_pairs)}** tokens  •  encoding: `{get_encoder().name}`")

    # ── 5b. Call Azure OpenAI ────────────────────
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

    # Full reply
    st.markdown(f"> {reply_text}")

    # Usage stats
    if response.usage:
        u = response.usage
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt tokens", u.prompt_tokens)
        col2.metric("Completion tokens", u.completion_tokens)
        col3.metric("Total tokens", u.total_tokens)

    # ── 5c. Log-prob analysis ────────────────────
    if logprobs_data and logprobs_data.content:
        tokens_info = logprobs_data.content

        # Collect data for table & chart
        table_rows = []
        chart_tokens = []
        chart_probs = []

        for idx, ti in enumerate(tokens_info):
            pct = logprob_to_pct(ti.logprob)
            alts = ", ".join(
                f"{a.token!r} ({logprob_to_pct(a.logprob):.1f}%)"
                for a in (ti.top_logprobs or [])
                if a.token != ti.token
            )
            table_rows.append(
                {
                    "#": idx + 1,
                    "Token": repr(ti.token),
                    "Log Prob": f"{ti.logprob:.4f}",
                    "Probability %": f"{pct:.1f}",
                    "Top Alternatives": alts or "—",
                }
            )
            chart_tokens.append(ti.token)
            chart_probs.append(pct)

        # ── Response tokens as colour-coded badges ──
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("🎨  Response Tokens — Colour-Coded by Confidence")
        st.markdown(
            render_token_badges(
                chart_tokens,
                probabilities=chart_probs,
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "🟩 >90 %  · 🟨 50–90 %  · 🟧 10–50 %  · 🟥 <10 %"
        )

        # ── Table ───────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("📊  Token-by-Token Log Probabilities")
        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
        )

        # ── Overview bar chart ──────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("📈  Probability Overview Chart")

        bar_colors = [
            "#388e3c" if p > 90 else
            "#f9a825" if p > 50 else
            "#e65100" if p > 10 else
            "#c62828"
            for p in chart_probs
        ]

        fig = go.Figure(
            go.Bar(
                x=list(range(1, len(chart_tokens) + 1)),
                y=chart_probs,
                text=[repr(t) for t in chart_tokens],
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
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Per-token detail expanders ──────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("🔍  Token Details — Explore Alternatives")
        st.caption("Click any token to see what else the model considered.")

        for idx, ti in enumerate(tokens_info):
            pct = logprob_to_pct(ti.logprob)
            label = f"Token {idx + 1}:  {ti.token!r}  —  {pct:.1f} %"
            with st.expander(label):
                if ti.top_logprobs:
                    alt_tokens = [a.token for a in ti.top_logprobs]
                    alt_probs = [logprob_to_pct(a.logprob) for a in ti.top_logprobs]

                    fig_alt = go.Figure(
                        go.Bar(
                            y=[repr(t) for t in alt_tokens],
                            x=alt_probs,
                            orientation="h",
                            marker_color=[
                                "#388e3c" if t == ti.token else "#90a4ae"
                                for t in alt_tokens
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
                    )
                    st.plotly_chart(fig_alt, use_container_width=True)
                else:
                    st.info("No alternative tokens returned for this position.")
    else:
        st.warning("The model did not return log probability data.")

elif submit:
    st.warning("Please enter a prompt first.")
