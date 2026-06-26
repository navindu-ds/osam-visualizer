## What the memory grid is

- The grid is the **OSAM state matrix** $S$: a fixed $r \times r$ table (default 8×8) that stores all accumulated writes.
- Size never grows with more sentences — this is the core property from δ-mem (Lei et al., 2026).
- Each cell holds a signed real value; meaning is **distributed** across the matrix, not stored in one cell per word.
- The matrix starts at zero; each **Insert** applies one delta-rule update.

## How to use the UI

- **Insert** — encode the sentence, project to key/value vectors, update $S$. The live heatmap refreshes; changed cells may pulse briefly.
- **Retrieve** — encode the query, read $r = S q$, rank stored sentences by cosine similarity to $r$. Memory is **not** modified.
- **Sentence registry** — click a sentence to inspect that write step: net change or component views. Use **Back to Live View** to return to current $S$.
- **Component radio** — switch among Net Change, Retention, Erase Prediction, Write New Value, and Final State ($S_t$). Selection persists when you click other sentences.
- **Parameters** — $r$ (matrix size, requires reset), $\beta$ (write strength), `top_k` (retrieval results). Changing $r$ after writes triggers a reset confirmation.
- **Reset** — clears $S$, registry, and write history.

## How to read the heatmaps

- **Colors** — diverging scale: blue = negative, white ≈ 0, red = positive.
- **Legend** — fixed ± scale across live view, all sentences, and all components (based on max |value| over the session).
- **Live view** — current consolidated memory state $S$.
- **Net Change** — $S_t$ - $S_{t-1}$ for the selected sentence step (what changed in that update).
- **Component views** — the three terms that **add up to** the new memory state after new sentence $x_t$ is added is shown by $S_t$ (retention, erase prediction, write new value), plus Final State. 
- **First sentence** — Retention and Erase are zero (no prior state); only Net Change, Write New Value, and Final State apply.

## Per-sentence views (important)

- Clicking a sentence shows the update at that moment — useful and exact for that step.
- It is **not** an isolated, order-independent fingerprint of one sentence; later writes depend on earlier state.

## Similarity scores

| Score | Interpretation |
|-------|----------------|
| **+1** | Readout aligns strongly with that sentence's stored value vector |
| **0** | No directional overlap |
| **−1** | Opposite direction in vector space |

**Demo caveat:** projections are fixed and untrained, so treat high scores as "related" and near-zero as "unrelated". Negative scores are often geometric, not semantic disagreement.

## Glossary

| Term | Meaning |
|------|---------|
| Memory state / grid | Fixed $r \times r$ matrix $S$ |
| Write / Insert | Delta-rule update from a new sentence |
| Read / Retrieve | Query $S$ without updating it |
| Vector | Numeric representation of text after encoding and projection |
| $\beta$ (beta) | Write strength — balance of retention vs. new write |
| Similarity score | Cosine similarity between readout and a stored value vector |
