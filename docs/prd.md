## Project overview

- Standalone implementation of the OSAM (Online State of Associative Memory) module from δ-mem (Lei et al., 2026 -- [https://arxiv.org/abs/2605.12357](https://arxiv.org/abs/2605.12357))
- Demonstrates delta-rule memory update and retrieval mechanics without a full LLM backbone
- Delivered as a hosted interactive web application (Streamlit)
- Scope: ~3 day concentrated effort; green-field Python codebase

## Functional Requirements

- User can input text sentences one at a time to be written into memory core
- Each sentence is encoded via a local sentence encoder (all-MiniLM-L6-v2, no API key required)
- Memory state matrix is updated after each sentence using the SSW (Sequence-State Write) strategy 
    - updates after a messages sequence is complete - so a collection of tokens that forms a single message
- User can submit a query to retrieve the most relevant stored sentence
- Retrieval uses OSAM read (Eq. 6) followed by cosine similarity against a sentence registry
- Results display ranked list of stored sentences with similarity scores
- Memory state heatmap updates live after each write operation core
- User can reset the memory state and registry to start fresh

## Configurable parameters

> Ensure to verify these parameters from the original paper to understand and verify the behaviour of the parameters and the effect made

- Memory size r — dimension of the square state matrix (default: 8)
- Write strength β — controls how strongly new info is written (default: 0.1)
- Retention gate λ — derived as 1 − β; controls how much old memory is kept
- Max input length — character cap per sentence (default: 500)
- Max stored sentences — upper limit on registry size (default: 30)

## Non-functional requirements

- All OSAM math in pure NumPy — no ML framework required
- Sentence encoder cached on startup via @st.cache_resource
- Session state is per-user; no data shared between sessions
- App functional on free-tier Streamlit Cloud CPU
- Dependencies pinned in requirements.txt
- All config values in a single config.py or defaults.yaml
- Codebase must pass a linter (flake8 or ruff)
- Docstrings on all public functions citing paper equations

## Project Structure

- src/memory_state.py — OSAM core: read, write, delta-rule update
- src/writing_strategies.py — SSW implementation
- src/encoder.py — sentence encoder wrapper
- src/retrieval.py — registry management and cosine similarity scoring
- config/defaults.yaml — all hyperparameters and constants
- tests/ — unit tests (pytest) for each src module
- app.py — Streamlit UI entry point
- README.md — setup, usage, architecture overview, paper reference

## Testing requirements

- Unit tests for delta-rule update with known numerical inputs and expected outputs
- Unit tests for OSAM read operation — correct output shape and values
- Unit tests for Sequence State Writing (SSW) sentence averaging
- Unit tests for input validation — oversized inputs rejected correctly
- One end-to-end integration test: write 5 sentences → query → assert top result is correct

## Out of scope

> Not required to completed

- LLM backbone integration or attention correction (Eq. 7–9)
- TSW and MSW writing strategies
- Model training or fine-tuning of projection weights
- Multi-user persistence or database storage
- Authentication or access control