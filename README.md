# OSAM Memory Visualizer

[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.12357)
[![Demo](https://img.shields.io/badge/Demo-Live-FF4B4B?logo=streamlit&logoColor=white)](https://osam-visualizer.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/navindu-ds/osam-visualizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive **Streamlit** web app for exploring the **Online State of Associative Memory (OSAM)** update and read mechanics described in [δ-mem: Efficient Online Memory for Large Language Models](https://arxiv.org/abs/2605.12357) (Lei et al., 2026).

Write sentences into a fixed-size memory matrix, inspect how each write updates the state, and test read-only retrieval — all with live heatmaps and per-sentence change views.

> **Not the official δ-mem implementation.** This repo is an independent educational demo. It implements the core OSAM **delta-rule write** and **matrix read**, but it does **not** integrate with or steer LLM weights, attention, or text generation. Hidden states come from a lightweight **SentenceTransformer** encoder (not LLM layer outputs), projections are **fixed and untrained**, and retrieval is a **cosine-similarity lookup** over a sentence registry — not the paper's low-rank attention correction pathway into an LLM. Use it to visualize memory dynamics, not to reproduce δ-mem end-to-end results.

---

## About the paper

**δ-mem** introduces a compact, online memory for LLMs: a fixed **r × r** state matrix **S** updated by a delta rule and read with a single matrix–vector product. Memory cost stays **O(r²)**, independent of conversation length.

| | |
|---|---|
| **Title** | δ-mem: Efficient Online Memory for Large Language Models |
| **Authors** | Jingdi Lei, Di Zhang, Junxian Li, Weida Wang, Kaixuan Fan, Xiang Liu, Qihan Liu, Xiaoteng Ma, Baian Chen, Soujanya Poria |
| **Year** | 2026 |
| **Paper** | [arXiv:2605.12357](https://arxiv.org/abs/2605.12357) |

This repository is an **educational visualization** of the OSAM write/read mechanics. It is not the official δ-mem implementation.

---

## Features

- **Live memory heatmap** — color-coded **r × r** matrix with write-pulse animation
- **Insert & Retrieve** — Sequence-State Write (SSW); read via `r = S q` and cosine ranking
- **Per-sentence inspection** — click the sentence registry to view net change `S_t − S_{t−1}`
- **Component decomposition** — Retention, Erase Prediction, Write New Value, and Final State `S_t`
- **Unified color scale** — shared legend across live view, all steps, and components
- **In-app guide** — ℹ **How it works** modal (Guide + Technical tabs, diagrams, LaTeX)
- **Configurable parameters** — matrix size **r**, write strength **β**, top-k retrieval

---

## Demo vs. paper

| Aspect | Paper (δ-mem) | This demo |
|--------|---------------|-----------|
| Hidden state `x_t` | LLM layer output | SentenceTransformer embedding |
| Projections `W_k, W_v, W_q` | Trained | Fixed random init (seed 42) |
| Write gate **β** | Per-row sigmoid | Scalar UI slider |
| Delta update & read | Exact OSAM | **Faithful** |
| Output | Low-rank attention → LLM text | Registry cosine lookup |
| Writing strategies | TSW / SSW / MSW | **SSW only** |

Retrieval scores are **illustrative** (untrained projections). Session state is **not persisted** across browser refreshes.

---

## Project structure

```
osam-visualizer/
├── app.py                 # Streamlit application entry point
├── config/
│   └── defaults.yaml      # Memory, encoder, UI, and viz defaults
├── src/                   # OSAM core (NumPy)
│   ├── memory_state.py    # Delta-rule update & read
│   ├── encoder.py         # SentenceTransformer + projections
│   ├── writing_strategies.py
│   └── retrieval.py       # Sentence registry & ranking
├── ui/                    # Streamlit UI components
│   ├── info_dialog.py     # How-it-works modal
│   └── header_badges.py
├── docs/                  # Methodology, specs, info content, assets
├── tests/                 # pytest unit & integration tests
└── requirements.txt
```

---

## Requirements

- **Python** 3.10 or newer (3.11+ recommended)
- **pip**
- Internet access on first run (downloads `all-MiniLM-L6-v2` from Hugging Face)

---

## Installation (local)

### 1. Clone the repository

```bash
git clone https://github.com/navindu-ds/osam-visualizer.git
cd osam-visualizer
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first app launch downloads the sentence encoder (~90 MB). Allow a minute on a slow connection.

---

## Run locally

From the repository root:

```bash
streamlit run app.py
```

Streamlit opens the app at **http://localhost:8501** by default.

### Quick start in the UI

1. **Insert** a few sentences and watch the memory matrix update.
2. Click a sentence in the **registry** to inspect that write step (net change or components).
3. **Retrieve** with a query to see ranked similarity scores (read-only).
4. Press **ℹ** (top right) for the full guide and technical notes.
5. **Reset** to clear memory and history.

---

## Configuration

Defaults live in [`config/defaults.yaml`](config/defaults.yaml). Common settings:

| Key | Default | Description |
|-----|---------|-------------|
| `memory.size` | `8` | Matrix dimension **r** (requires reset if changed after writes) |
| `memory.write_strength` | `0.1` | Default **β** (write strength) |
| `retrieval.top_k` | `5` | Number of retrieval results |
| `visualization.matrix_decimal_places` | `4` | Heatmap cell precision |
| `encoder.model_name` | `all-MiniLM-L6-v2` | Sentence embedding model |

Streamlit theme and server options: [`.streamlit/config.toml`](.streamlit/config.toml).

---

## Development

### Run tests

```bash
pytest
```

### Lint

```bash
ruff check app.py src/ ui/ tests/
```

### GitHub Codespaces

A [`.devcontainer`](.devcontainer/devcontainer.json) is included. Open the repo in Codespaces; dependencies install automatically and Streamlit starts on port **8501**.

---

## Deployment

**Live demo:** [https://osam-visualizer.streamlit.app/](https://osam-visualizer.streamlit.app/)

The app is compatible with [Streamlit Community Cloud](https://streamlit.io/cloud):

- **Main file:** `app.py`
- **Python version:** 3.10+
- **Requirements:** `requirements.txt`

`fileWatcherType = "none"` is set for stable cloud hosting.

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/info/guide.md`](docs/info/guide.md) | User guide (also shown in the ℹ modal) |
| [`docs/info/technical.md`](docs/info/technical.md) | Technical notes & limitations |
| [`docs/paper/osam_methodology.md`](docs/paper/osam_methodology.md) | OSAM methodology mapped to the paper |
| [`docs/functionality.md`](docs/functionality.md) | Backend data flow |
| [`spec.md`](spec.md) | Full project specification |

---

## Citing the original paper

This repository is **not** affiliated with the δ-mem authors. If your work discusses, builds on, or compares against **δ-mem / OSAM** itself, cite the **original paper** (not this demo):

```bibtex
@article{lei2026deltamem,
  title   = {$\delta$-mem: Efficient Online Memory for Large Language Models},
  author  = {Lei, Jingdi and Zhang, Di and Li, Junxian and Wang, Weida and Fan, Kaixuan and Liu, Xiang and Liu, Qihan and Ma, Xiaoteng and Chen, Baian and Poria, Soujanya},
  year    = {2026},
  journal = {arXiv preprint arXiv:2605.12357},
  url     = {https://arxiv.org/abs/2605.12357}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

The δ-mem paper and its methods are the work of the original authors; this repository is an independent educational implementation.

---

## AI-assisted development

**GitHub Copilot** and **Cursor** were used to generate code and documentation in this repository. The developer planned features with AI assistance, reviewed and validated each plan before implementation, and then used AI to produce code from those approved plans.

All generated output was **manually reviewed and validated** by the developer. Testing and code review were performed **both manually and with AI assistance**. The developer remains responsible for the final implementation choices and correctness of the published code.
