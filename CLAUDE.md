# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app that classifies text sentiment as positive/negative using SiEBERT (`siebert/sentiment-roberta-large-english`). Users upload a CSV (or try built-in sample data), select the text column, classify, and download results with "Sentiment" and "Confidence" columns. Guided step-by-step UI with auto-detected text columns and summary metrics.

## Commands

```bash
# Setup
uv sync

# Run
uv run streamlit run streamlit_app.py

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check .

# Test
uv run pytest                                          # all tests
uv run pytest tests/test_streamlit_app.py              # single file
uv run pytest tests/test_streamlit_app.py::test_name   # single test
```

Use `ruff` for all linting and formatting. Run `uv run ruff check --fix .` to auto-fix lint issues. Use `ty` for type checking. Use `pytest` for unit testing.

## Architecture

Single-file application (`streamlit_app.py`, ~185 lines):

1. **`detect_text_column`** — returns first string-dtype column name via `next()` generator
2. **`load_model`** — loads config via `AutoConfig`, constructs `RobertaForSequenceClassification`, loads weights via `from_pretrained` with `float16=True`; cached with `@st.cache_resource`; authenticates with `HF_TOKEN`
3. **`process_dataframe`** — pre-filters blanks, batches valid texts (`BATCH_SIZE=8`), tokenizes with `return_tensors="np"` and converts to `mx.array`, classifies via softmax over logits; uses `.tolist()` for batch conversion
4. **UI** — guided step-by-step flow: file upload or sample data → column auto-detect and preview → classify → summary metrics → results table → CSV download

## Key Patterns

- MLX for all inference on Apple Silicon (no device management needed)
- `hf_logging.set_verbosity_error()` suppresses expected checkpoint warnings
- Confidence via `mx.softmax(logits, axis=-1)` with `mx.max` and `mx.argmax`; `mx.eval()` before `.tolist()`; labels from `model.config.id2label`
- Empty/whitespace-only texts skipped; get sentiment `""` and confidence `0.0`
- Tokenizer uses `return_tensors="np"` converted to `mx.array`, with `truncation=True` (512 token limit) and `padding=True`
- `process_dataframe` returns a copy; input DataFrame is not mutated
- `st.session_state` persists loaded DataFrame across Streamlit reruns (buttons reset on rerun)
- Walrus operator (`:=`) in UI guards to combine detect + check into one `elif`
- `SAMPLE_DATA_PATH` points to `tests/data/csv/mixed_sample.csv` for the "Try with sample data" button
- Uses Streamlit default theme settings (no custom `.streamlit/config.toml`)
- Dependencies managed by `uv` with lockfile (`uv.lock`)

## Tests

- `tests/conftest.py` — module-level patch for `RobertaForSequenceClassification` to prevent model downloads during test collection
- `tests/test_streamlit_app.py` — unit tests for `detect_text_column`, `load_model`, `process_dataframe`, `BATCH_SIZE`, and `SAMPLE_DATA_PATH`; uses `autouse` fixture for Streamlit mock and mocked model/tokenizer throughout
- `tests/data/csv/product_reviews.csv` — 40 e-commerce product reviews
- `tests/data/csv/movie_reviews.csv` — 40 film and TV opinions
- `tests/data/csv/social_media.csv` — 40 tweets and social media posts
- `tests/data/csv/restaurant_reviews.csv` — 40 dining and food service reviews
- `tests/data/csv/app_reviews.csv` — 40 mobile/web app store reviews
- `tests/data/csv/mixed_sample.csv` — 20-row sample (4 from each domain) for quick testing
