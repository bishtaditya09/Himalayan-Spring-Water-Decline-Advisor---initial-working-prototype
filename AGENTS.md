# AGENTS.md (Agent/Coding Mode)

This file provides guidance to agents when working with code in this repository.

## Non-obvious coding constraints

- **Bare imports only in `app.py`**: `src/` is injected into `sys.path` at runtime. Import as `from forecasting import …`, never `from src.forecasting import …`.
- **Corpus `.txt` format is load-sensitive**: `_load_corpus()` splits on the first `\n\n` to separate title from body, then splits body on `\n\n` for paragraph chunks. Paragraphs shorter than 40 characters are silently dropped. New corpus files must follow `Title: <name>\n\n<body paragraphs separated by blank lines>`.
- **`SpringAdvisoryRAG` is a stateful object** — `TfidfVectorizer` is fit once in `__init__`. If corpus changes, instantiate a new `SpringAdvisoryRAG`; do not call `fit_transform` again on the existing instance.
- **`forecast_spring()` uses `pd.infer_freq()`** on the DatetimeIndex — input data must be strictly monthly (`freq="MS"`). Gaps in the CSV will cause `infer_freq` to return `None` and break Holt-Winters initialization.
- **Risk threshold literals are duplicated in UI**: `_classify_risk()` in `forecasting.py` and the `risk_colors` dict in `app.py` both enumerate `"CRITICAL"/"WATCH"/"STABLE"`. Changing a level name requires updating both.
- **`forecast_vals.clip(lower=0.1)`** — always preserve this; Holt-Winters can produce negative forecasts for declining series near zero, which would break log-scale plots and metric displays.
- **Streamlit cache invalidation**: `load_data()` uses `@st.cache_data` keyed on function signature. Changing `DATA_PATH` or adding params requires clearing cache (`st.cache_data.clear()`) or the old data will persist in the session.
