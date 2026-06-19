# OSAM Replication Methodology — δ-mem (Lei et al., 2026)

Scope note: this extracts only the **Online State of Associative Memory (OSAM)** core —
the memory matrix, its delta-rule update, and its read mechanism. The attention-steering
/ low-rank correction part (Section 3.3, query/output injection into a frozen backbone)
and the SFT training loop are **out of scope** for this visualization tool, but are listed
briefly at the end for context since the paper frames OSAM as feeding into them.

---

## 1. Core Object: The Memory State

- **State matrix `S`**: shape `r × r` (paper uses `r = 8`, i.e. an 8×8 matrix). This is the
  entire "memory" — no token list, no growing KV cache. Fixed size regardless of sequence length.
- `S` is initialized at zero (or near-zero) before any tokens are processed: `S_0 = 0`.
- `S` is updated once per "write event" (granularity choice — see Section 4 below) and persists
  across the whole interaction/session.
- Everything happens in a single low-dimensional projected space of dimension `r`, **not** in
  the full hidden dimension `d` of an LLM. This is what makes it cheap and visualizable.

## 2. Projecting Inputs into Memory Space

For each incoming token/segment with hidden vector `x_t` (dimension `d`):

- **Query** (for reading): `q_t = L2norm(tanh(W_q · x_t))` → shape `r`
- **Key** (for writing): `k_t = L2norm(tanh(W_k · x_t))` → shape `r`
- **Value** (for writing): `v_t = W_v · x_t` → shape `r` (no normalization/activation)
- **Write gate**: `β_t = sigmoid(W_β · x_t + b)` → shape `r` (vector, one scalar per *row* of `S`)
- **Retention/forget gate**: `λ_t = 1 − β_t` → shape `r`

Replication note: since this is a **standalone visualization tool** (not wrapped around a real
LLM), `x_t` does not need to come from a real transformer hidden state. Substitute with:
- a simple learned/random embedding of input tokens, or
- a toy deterministic embedding (e.g. hash-based or character n-gram vector) projected to
  dimension `d`, then through `W_q, W_k, W_v, W_β` (random fixed matrices, or simple trainable
  linear layers) down to `r`.
The point of the visualizer is to show **state dynamics**, not to faithfully reproduce a
pretrained LLM's hidden states.

## 3. Reading from Memory (before writing)

- `r_t = S_{t-1} @ q_t` — read vector, shape `r`. Just a matrix-vector product against the
  **previous** state (read happens before the current write).
- This is the "query memory" operation users will trigger interactively: feed in a query vector,
  multiply against current `S`, return `r_t`.
- Cost is `O(r²)`, independent of how much history has been written — a key property to
  visualize/demonstrate (constant-time retrieval regardless of sequence length).

## 4. Writing into Memory (delta-rule update)

This is the heart of the mechanism — implement exactly as specified:

- **Prediction error formulation**: treat the update as one step of online SGD minimizing
  `L_t(S) = ½‖S k_t − v_t‖²`, giving the un-gated delta rule:
  `S_t = S_{t-1} + β_t (v_t − S_{t-1} k_t) k_t^T`
- **Gated version actually used** (dimension-wise gating, applied **per row** of `S`):
  ```
  S_t = diag(λ_t) @ S_{t-1} + diag(β_t) @ (v_t − S_{t-1} @ k_t) @ k_t^T
  ```
  Equivalently, expanded into three interpretable terms:
  1. `diag(λ_t) @ S_{t-1}` — retain old state (decayed per-row)
  2. `− diag(β_t) @ S_{t-1} @ k_t @ k_t^T` — erase old prediction along key direction `k_t`
  3. `+ diag(β_t) @ v_t @ k_t^T` — write new value along key direction `k_t`
- **Row-wise view** (useful for visualizing which memory rows change and by how much):
  for row `i`: `s_t^(i) = λ_t,i · s_{t-1}^(i) + β_t,i · (v_t,i − s_{t-1}^(i) · k_t) · k_t^T`
- Implementation detail: `β_t` and `λ_t` are **vectors** of length `r` (one gate value per row),
  not scalars — so each row of the state matrix can independently decide how much to retain vs.
  overwrite. This is exactly the signal worth visualizing (e.g. a heatmap of gate values per row
  alongside the state matrix heatmap).

## 5. Writing Granularity (pick at least one; SSW or TSW are simplest to implement first)

- **Token-State Write (TSW)**: call the write update at every single token. Finest granularity,
  most reactive to noise/formatting tokens. Simplest to implement first.
- **Sequence-State Write (SSW)**: average hidden states over all tokens in a "message" (i.e.
  one user turn / one chunk of input text), then do **one** write per message using the averaged
  vector `x̄^(j) = mean(x_t for t in message_j)`. Smoother state evolution — good default for a
  chat-style demo UI where one "write" = one submitted message.
- **Multi-State Write (MSW)**: maintain `N` parallel `r × r` sub-states (paper uses `N = 4`), each
  updated independently with its own gate/projection parameters, then concatenate their read
  vectors `r_t = concat(r_t^(1), ..., r_t^(N))` for output. More complex; treat as a stretch goal
  / toggle-able mode, not the MVP.

## 6. What to Actually Visualize (interactive demo design)

Map paper concepts to UI elements:

- **State matrix heatmap**: render `S` (r×r, e.g. 8×8) as a color-coded grid; update live after
  each write. This is the single most important visual.
- **Diff/delta highlight**: on each write, show `S_t − S_{t-1}` as a secondary heatmap or
  overlay (highlights which cells changed and by how much) — directly visualizes the "residual
  write" property described in the paper (well-learned associations → small updates).
- **Gate values bar/strip**: show `β_t` and `λ_t` (length-r vectors) as small bar charts per row,
  so users see *why* some rows changed more than others.
- **Read vector display**: when a user submits a "query", show `q_t`, then `r_t = S @ q_t`, and
  optionally decode/project `r_t` back to something interpretable (e.g. nearest stored
  key-value pair by cosine similarity, for demo purposes).
- **Sequential input log**: a running list of processed inputs (tokens or messages) with which
  write granularity produced which state — lets users scrub back/forward through state history
  if you store `S_t` snapshots.
- **Constant-size emphasis**: explicitly show that `S` never grows regardless of how many inputs
  have been processed — e.g. a counter of "tokens/messages processed" next to the fixed 8×8 grid,
  to make the compression property visually obvious.
