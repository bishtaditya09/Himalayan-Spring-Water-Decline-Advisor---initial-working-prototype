# AGENTS.md (Plan Mode)

This file provides guidance to agents when working with code in this repository.

## Non-obvious architectural constraints

- **Two completely separate AI subsystems sharing only `df`**: forecasting (`src/forecasting.py`) and RAG (`src/rag_pipeline.py`) are independent — they share no state, no model, no data path beyond the Streamlit app layer.
- **No persistence layer for model state**: `SpringAdvisoryRAG` is reconstructed each Streamlit session (via `@st.cache_resource`). `forecast_spring()` is reconstructed each widget interaction. There is no saved model file, no pickle, no database.
- **District overview (`forecast_district_summary`) is O(n_springs) model fits**: fitting 6 Holt-Winters models per district selection. Any plan to add more springs per district will linearly increase UI wait time for Tab 1.
- **Swap point for neural retrieval is `embed_with_sentence_transformers_STUB()`** in `rag_pipeline.py`. Architecture is designed for drop-in replacement: swap `TfidfVectorizer`/`cosine_similarity` with `SentenceTransformer.encode()` + FAISS index in `__init__` and `retrieve()` only — `answer()` interface is stable.
- **Data generation is deterministic via `RNG_SEED = 42`**: re-running `generate_dataset()` with the same seed produces identical CSV. Changing `DISTRICTS` dict or generation logic changes all downstream forecasts and risk classifications — plan for full re-validation of expected risk level outputs.
- **No test suite exists**: all module entry points use `if __name__ == "__main__"` smoke tests. Any plan to add automated tests must account for Streamlit-specific code in `app.py` not being testable with standard pytest without mocking `st.*`.
