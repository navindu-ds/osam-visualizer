# Software Engineering Guidelines (Quick Reference)

- Main goals are maintainability to improve ease of handling code and documentation

- **One module, one responsibility** — `memory_state.py` only does OSAM math, no I/O or UI code
- **No magic numbers** — all constants (r, β, λ, limits) live in `config.py` / `defaults.yaml`
- **Type hints everywhere** — every function signature, no exceptions
- **Docstrings cite the paper** — e.g. `"""Delta-rule update (δ-mem Eq. X)."""`
- **Pure functions for math** — state update/read functions take inputs, return outputs, no hidden side effects
- **Test the math, not the UI** — pytest unit tests for every core function with known input → expected output
- **Validate inputs early** — length/empty checks at the entry point, fail with clear errors
- **Cache expensive resources** — encoder model loaded once (`@st.cache_resource`), never per-request
- **No hardcoded secrets/keys** — none needed, but keep it that way
- **Small, named functions over long scripts** — if a function exceeds ~30 lines, split it
- **Consistent naming matching the paper** — `S`, `k_t`, `v_t`, `q_t`, `beta`, `lambda_` — makes code traceable to math
- **README explains setup + architecture + paper mapping** — assume the next reader has never seen the code
- **Lint before done** — run `ruff`/`flake8`, fix warnings, not just errors
- **Commit small, commit often** — clear commit messages, not one giant commit