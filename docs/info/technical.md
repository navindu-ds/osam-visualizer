## Core objects

- **State matrix** $S \in \mathbb{R}^{r \times r}$ — OSAM memory; initialized to zero.
- **Hidden vector** $x_t \in \mathbb{R}^d$ — sentence embedding from SentenceTransformer (`all-MiniLM-L6-v2`, $d=384$).
- **Projections** (fixed random matrices, seed 42):
  - $k_t = \mathrm{L2norm}(\tanh(W_k x_t))$
  - $v_t = W_v x_t$
  - $q_t = \mathrm{L2norm}(\tanh(W_q x_{\mathrm{query}}))$ (retrieval only)
- **Write gate** — scalar $\beta \in (0, 1]$ from UI slider (paper uses per-row $\mathrm{sigmoid}(W_\beta x + b)$).

## Write (SSW — one update per sentence)

Sequence-State Write: one encoder pass and one OSAM update per submitted sentence.

## Read

- $r_t = S q_t$ — matrix–vector product; cost $O(r^2)$, independent of history length.
- Demo output: $\mathrm{score}(i) = \mathrm{cosine}(r_t, v_i)$ over sentence registry value vectors; top-k shown in UI.

## Scope boundaries

- No frozen LLM backbone; no low-rank attention correction pathway.
- Registry + cosine ranking substitutes for LLM text generation so read direction is observable.
- TSW and MSW writing strategies are not implemented.

## Limitations

- Untrained projection matrices — retrieval quality is illustrative, not production-grade.
- Scalar $\beta$ — all rows share the same write strength (paper gates per row).
- Order-dependent consolidation — write order affects final $S$.
- Similarity scores reflect geometry in projected space, not guaranteed semantic entailment.
- Session-only state — no persistence across browser sessions.

## References

- Lei et al., 2026 — [δ-mem: Efficient Online Memory for Large Language Models](https://arxiv.org/abs/2605.12357)
- Methodology notes — `docs/paper/osam_methodology.md`
- Backend flow — `docs/functionality.md`
