# Backend & Functionality Specification — OSAM Memory Visualizer

## 1. Purpose of This Document

This document describes the internal data flow, mathematical operations, and state
management logic that run behind the UI. It explains, step by step, what happens when a
sentence is written to memory and when a query is retrieved, how each step maps to the
methodology in the δ-mem paper (Lei et al., 2026), and how session data is held and
reset. It is intended as the technical companion to the UI Specification and the PRD.

---

## 2. Scope Recap

As defined in the PRD, this implementation isolates the **Online State of Associative
Memory (OSAM)** component described in the paper and runs it independently of the frozen
LLM backbone. The paper's low-rank attention correction pathway (which requires a live
Transformer forward pass) is out of scope. Within that boundary, the implementation
follows the paper's OSAM mechanics exactly: the state matrix, the delta-rule write, and
the read operation are reproduced faithfully; only the surrounding system (LLM hidden
states in, attention correction out) is substituted with components suited to a
standalone demo.

---

## 3. Core Data Structures

Three pieces of state are held per user session:

| Component | Type | Description |
|---|---|---|
| Memory state `S` | `r × r` NumPy array | The OSAM matrix itself (paper's core object) |
| Sentence registry | List of `(text, key_vector, value_vector, step_index)` | Side store mapping original sentences to their projected vectors, used for retrieval scoring |
| Step counter | Integer | Tracks how many sentences have been written, shown in the UI |

`r` is the configurable memory size (default 8, per the paper's reported 8×8 setting).

---

## 4. The Write Process (Insert)

This is triggered when the user submits a sentence via the Insert button. Each numbered
step below corresponds to a specific stage in the paper's methodology.

### Step 4.1 — Text to Hidden State

The input sentence is passed through a local sentence encoder to produce a dense vector
`x_t ∈ R^d`.

*Paper correspondence:* in δ-mem, `x_t` is the hidden state emitted by a layer of the
frozen Transformer backbone as it processes the token/segment. Since no LLM is used here,
the sentence encoder's output vector plays the same structural role: a fixed-dimensional
representation of "what was just said," which is what δ-mem's projections operate on.

### Step 4.2 — Projection into Memory Space

`x_t` is projected into three vectors using fixed projection matrices:

- Key: `k_t = normalize(tanh(W_k · x_t))`
- Value: `v_t = W_v · x_t`
- (Query projection `W_q` is defined here too, but only used at retrieval time — see
  Section 5.1)

*Paper correspondence:* Section 3.1, the projection of backbone hidden states into the
low-dimensional memory space prior to writing. The paper learns these projections during
training; in this standalone implementation they are fixed (randomly initialized once at
app startup and held constant), since no training loop exists in this scope. This
substitution is documented as a stated limitation, not hidden.

### Step 4.3 — Write Gate Computation

A scalar write strength `β_t` is computed (or taken directly from the configured
parameter, depending on whether per-step adaptivity is enabled). Its complement
`λ_t = 1 − β_t` is the retention gate.

*Paper correspondence:* Section 3.3, the gating mechanism that balances how much new
information overwrites the state versus how much old information is retained.

### Step 4.4 — Delta-Rule Update

The state matrix is updated as:

```
S_t = λ_t · S_{t-1}  +  β_t · (v_t − S_{t-1} · k_t) · k_t^T
```

Plainly: `S_{t-1} · k_t` is what the memory currently "predicts" along this key
direction; the difference `v_t − S_{t-1} · k_t` is the residual — the part of this fact
the memory doesn't already encode. Only that residual gets written in, scaled by `β_t`,
while the old state is scaled down by `λ_t`. This is the delta rule that gives the paper
its name, and is the central operation this project reproduces exactly.

*Paper correspondence:* the OSAM update equation (Section 3.3 / Eq. for state update).
This is implemented with no simplification — it is the most important piece of fidelity
to the source paper.

### Step 4.5 — Writing Strategy: SSW

Per the PRD, only **Sequence-State Write (SSW)** is implemented. Under SSW, if a
"sentence" is treated as a single segment (rather than updating per-token), the encoder's
sentence-level output is used directly as `x_t`, and one delta-rule update runs per
submitted sentence. This matches the paper's SSW strategy at the segment granularity,
while skipping the token-level (TSW) and multi-stream (MSW) variants, which are
explicitly out of scope.

### Step 4.6 — Registry Update

The original sentence text, its `k_t`, its `v_t`, and the current step index are appended
to the sentence registry. This registry is **not part of the paper's method** — it exists
solely so the demo can map a memory readout back to readable text (see Section 5.3). This
substitution is necessary because the paper's actual output pathway is a text-generating
LLM, which is outside this project's scope.

### Step 4.7 — UI Signal

The backend returns the updated matrix `S_t`, a diff against `S_{t-1}` (used to compute
which cells changed most, driving the write-pulse highlight in the UI), and the
incremented step counter.

---

## 5. The Retrieve Process (Query)

Triggered when the user submits a query via the Retrieve button. Retrieval is read-only:
it does not modify `S`.

### Step 5.1 — Query Projection

The query text is encoded the same way as Step 4.1, then projected using the query matrix:

```
q_t = normalize(tanh(W_q · x_query))
```

*Paper correspondence:* Section 3.2, "Reading from Online State of Associative Memory" —
the query projection precedes the read step.

### Step 5.2 — Memory Read

```
r_t = S_{t-1} · q_t
```

This is a direct matrix-vector product against the **current** memory state (no update
happens here). The result `r_t ∈ R^r` is the memory readout vector.

*Paper correspondence:* this is exactly the paper's read operation. The paper notes this
read's cost is independent of history length, since `S` is fixed-size regardless of how
many facts have been written — this property is preserved here and can be called out
explicitly in the demo as a key property of the architecture.

### Step 5.3 — Mapping the Readout to Text

Since `r_t` is a vector, not text, it cannot be "decoded" back into a sentence directly —
there is no inverse function from matrix space to language. Instead, `r_t` is compared
via cosine similarity against every `v_t` stored in the sentence registry:

```
score(i) = cosine_similarity(r_t, registry[i].value_vector)
```

The registry entries are ranked by score, and the top-ranked entry's original text is
what gets shown as the system's "answer."

*Relationship to the paper, stated plainly:* in δ-mem, `r_t` is not converted to text by
comparison — it is fed forward into the LLM as a low-rank correction to attention, and the
LLM generates the answer. This implementation does not have an LLM, so the comparison
against the registry is a substitute mechanism that makes the *direction* `r_t` points to
in memory space observable and interpretable, without claiming to replicate text
generation. This is the most significant scope boundary in the whole project and should
be stated explicitly during the demo walkthrough.

### Step 5.4 — Result Construction

A ranked list of `(sentence_text, score)` pairs is returned, capped at a fixed number of
results (e.g. top 5, or fewer if the registry has fewer entries). The UI receives this
list and renders it as described in the UI Specification (Section 4.3).

---

## 6. Session State and Reset Behaviour

### 6.1 Storage Model

All state — the matrix `S`, the sentence registry, and the step counter — lives **only in
server-side memory for the duration of the user's browser session** (Streamlit's
`st.session_state`). Nothing is written to disk or any database.

- Each browser session gets its own independent `S`, registry, and counter
- Refreshing the page does not necessarily clear state (Streamlit persists
  `session_state` across reruns within the same session), but closing the
  tab/browser or the session timing out does
- No data persists across different users or different visits — this is a deliberate
  design choice appropriate for a stateless demo tool, not a production memory system

### 6.2 Reset Process

Triggered by the Reset control in the UI (Section 4.4 of the UI Spec). On confirmation:

1. `S` is reinitialized to a zero matrix of the currently configured size `r × r`
2. The sentence registry is cleared to an empty list
3. The step counter is reset to 0
4. The UI immediately reflects the empty state (Section 4.2 of the UI Spec)

### 6.3 Parameter Changes Mid-Session

If the user changes `r` (memory size) or `β` (write strength) after sentences have
already been written, this is treated as equivalent to a reset, since changing `r`
invalidates the existing matrix dimensions and changing `β` would make the existing state
inconsistent with new writes. The same confirmation flow applies.

---

## 7. Summary: Process-to-Paper Mapping

| Backend step | Paper section/equation | Faithful or substituted? |
|---|---|---|
| Text → hidden state | Backbone hidden state input | Substituted (sentence encoder replaces LLM layer) |
| Projection to k, v, q | Section 3.1 projections | Substituted (fixed weights, not trained) |
| Write gate β, λ | Section 3.3 gating | Faithful (formula preserved) |
| Delta-rule update | Section 3.3 OSAM update | **Faithful — exact** |
| SSW write strategy | Section on writing strategies | Faithful (TSW, MSW excluded by scope) |
| Memory read `r_t = S·q_t` | Section 3.2 reading | **Faithful — exact** |
| Readout → attention correction | Section 3.4 (low-rank correction) | Out of scope (no LLM) |
| Readout → text via registry lookup | — | Substituted (demo-only mechanism) |

This table is the single most important reference for the demo walkthrough: it lets the
presenter point to exactly which parts of the system are a direct, faithful
reimplementation of the paper's mathematics, and which parts are necessary, clearly
-justified substitutions made to keep the project standalone and within scope.