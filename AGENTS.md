# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Stack
Python 3.12, Streamlit, pandas, scikit-learn (TF-IDF), statsmodels (Holt-Winters), plotly. No test framework, no linter config.

## Run
```bash
# Must run from the spring-advisor/ directory (contains app.py)
cd spring-advisor
pip install -r requirements.txt
streamlit run app.py
```

## Module smoke-tests (no test runner — use __main__ blocks)
```bash
# From spring-advisor/ directory:
python src/data_generation.py   # regenerates data/springs_synthetic.csv
python src/forecasting.py        # prints one spring forecast to stdout
python src/rag_pipeline.py       # prints first 500 chars of a retrieved answer
```

## Critical architecture notes

- **`src/` is on `sys.path` via `app.py` line 8** (`sys.path.insert(0, …/src)`). Imports in `app.py` are bare (`from forecasting import …`), not `from src.forecasting import …`. Do not add `src.` prefixes.
- **`data/springs_synthetic.csv` is auto-generated on first run** if missing — `app.py`→`load_data()` calls `generate_dataset()` and writes it. Do not hardcode this path; use `os.path.join(os.path.dirname(__file__), "data", "springs_synthetic.csv")`.
- **`corpus/` path is resolved relative to `rag_pipeline.py`'s own `__file__`**, not the working directory. All corpus `.txt` files start with `Title: <name>` on line 1, then `\n\n`, then paragraph-chunked body.
- **No LLM, no API keys, no internet required.** RAG answers are TF-IDF + cosine similarity only. The `embed_with_sentence_transformers_STUB()` function in `rag_pipeline.py` is intentionally non-functional — do not call it.
- **`@st.cache_data` / `@st.cache_resource`** are used in `app.py` for `load_data()` and `load_rag()`. Mutations to cached DataFrames must use `.copy()` to avoid Streamlit cache mutation warnings.

## Code style (inferred from codebase — no linter enforced)
- Dataclasses (`@dataclass`) used for structured return types (`SpringForecast`, `Chunk`, `RetrievedResult`) — prefer this over dicts for new return types.
- Module-level constants in `ALL_CAPS` (`FORECAST_HORIZON_MONTHS`, `BASELINE_YEARS`, `RNG_SEED`, `CORPUS_DIR`).
- Paths always constructed with `os.path.join(os.path.dirname(os.path.abspath(__file__)), …)` — never hardcoded or relative to CWD.
- Risk levels are the string literals `"CRITICAL"`, `"WATCH"`, `"STABLE"` — used as dict keys and displayed directly in UI.
- `forecast_vals.clip(lower=0.1)` — forecasted discharge is always clipped to prevent negatives; maintain this if modifying the model.
- `generate_dataset(seed=RNG_SEED)` uses `np.random.default_rng(seed)` (new-style NumPy RNG). Do not mix with legacy `np.random.seed()`.

## Data disclosure (must preserve in UI)
The CSV data is synthetic. Any UI changes must retain the disclosure text — it is surfaced in the `st.expander` "About this project & data" and in the footer caption.
