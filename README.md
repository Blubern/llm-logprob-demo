# 🎲 Log Probability Explorer

An interactive Streamlit app that visualises **token-level log probabilities** from **OpenAI**, **Azure OpenAI (Azure AI Foundry)**, or **GitHub Models**.  
Built as an educational tool to help explain how Large Language Models generate text.

## Prerequisites

- Python 3.10+
- **One** of the following:
  - An [OpenAI API key](https://platform.openai.com/account/api-keys), **or**
  - An Azure AI Foundry / Azure OpenAI resource with a deployed chat model, **or**
  - A [GitHub Personal Access Token (PAT)](https://github.com/settings/tokens) for GitHub Models

## What It Does

1. **Tokenises your prompt** — shows exactly how the model splits your text into tokens, displayed as coloured badges.
2. **Sends the prompt** to an OpenAI or Azure-hosted model via the Chat Completions API with `logprobs` enabled.
3. **Displays the response** token-by-token with:
   - Colour-coded confidence badges (green → red)
   - A sortable data table with log probs, percentages, and top alternatives
   - An interactive Plotly bar chart of probabilities across the full response
   - Per-token expanders showing what other tokens the model considered

## Quick Start

### 1. Clone & install

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

Create (or edit) a `.env` file in the project root. Choose the section that matches your provider:

#### Option A — OpenAI

```dotenv
PROVIDER=openai
OPENAI_API_KEY=sk-...your-openai-api-key...
LOGPROB_MODEL=gpt-4o          # any model that supports logprobs
```

#### Option B — Azure OpenAI

```dotenv
PROVIDER=azure
OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
API_VERSION=2025-04-01-preview
LOGPROB_MODEL=gpt-5.2         # must match your Azure deployment name
```

#### Option C — GitHub Models

```dotenv
PROVIDER=github
GITHUB_TOKEN=ghp_...your-github-pat...
LOGPROB_MODEL=gpt-4o          # any model available on GitHub Models
```

To create a PAT, visit [github.com/settings/tokens](https://github.com/settings/tokens) and generate a token. If `GITHUB_TOKEN` is not set, the app falls back to `OPENAI_API_KEY`.

> **Notes:**
> - `PROVIDER` accepts `azure`, `openai`, or `github`. If omitted, the legacy `USE_AZURE` toggle is used for backward compatibility.
> - For **Azure**, the endpoint should be the *base* URL of your Azure Cognitive Services resource — the SDK builds the full Chat Completions path automatically.
> - For **OpenAI** / **GitHub Models**, `LOGPROB_MODEL` should be a model ID (e.g. `gpt-4o`, `gpt-4o-mini`). For **Azure**, it should match the **deployment name** in your Azure resource.
> - **GitHub Models** uses the endpoint `https://models.inference.ai.azure.com` which is handled automatically.

### 3. Run

```bash
python -m streamlit run logprob_demo.py
```

The app opens in your browser at `http://localhost:8501`.

## Understanding the Output

| Colour | Probability | Meaning |
|--------|------------|---------|
| 🟩 Green  | > 90 % | Model was very confident |
| 🟨 Yellow | 50 – 90 % | Fairly confident |
| 🟧 Orange | 10 – 50 % | Multiple plausible options |
| 🟥 Red    | < 10 % | Low confidence / surprising choice |

### Log Probabilities — Quick Primer

A **log probability** is the natural logarithm of the model's predicted probability for a token:

```
probability = e^(logprob)
```

- `logprob = 0.00` → 100 % confidence
- `logprob = −0.69` → ~50 %
- `logprob = −2.30` → ~10 %

The closer to zero, the more certain the model was.

## Project Structure

```
llm-logprob-demo/
├── .env                 # API credentials (not committed)
├── logprob_demo.py      # Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Tech Stack

- **[Streamlit](https://streamlit.io/)** — Web UI
- **[OpenAI Python SDK](https://github.com/openai/openai-python)** — Supports `OpenAI`, `AzureOpenAI`, and GitHub Models clients
- **[tiktoken](https://github.com/openai/tiktoken)** — BPE tokeniser for prompt visualisation
- **[Plotly](https://plotly.com/python/)** — Interactive charts
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** — `.env` file loading

## Screenshots

![Screenshot 1](images/Screenshot1.png)

![Screenshot 2](images/Screenshot2.png)

![Screenshot 3](images/Screenshot3.png)
