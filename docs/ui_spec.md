# UI Specification — OSAM Memory Visualizer

## 1. Purpose of This Document

This document specifies the layout, content, and interaction behaviour of the web
application's user interface. It defines what the user sees, what actions they can take,
and how the interface responds — independent of backend implementation details, which
are covered in the architecture/PRD documents.

---

## 2. Page Layout Overview

The application is a single-page interface with three persistent regions:

```
┌─────────────────────────────────────────────────────────┐
│                        Header / Title                     │
├───────────────────────────────┬───────────────────────────┤
│                                 │                           │
│         Main Panel             │     Sentence Registry     │
│   (Input, Matrix, Results)     │      (Side Panel)         │
│                                 │                           │
└───────────────────────────────┴───────────────────────────┘
```

- **Main panel** (left, ~65% width): all primary interaction — input box, action
  buttons, memory matrix heatmap, retrieval results.
- **Side panel** (right, ~35% width): always-visible registry of stored sentences.
- On narrow/mobile viewports, the side panel collapses below the main panel.

---

## 3. Header

- App title: "OSAM Memory Visualizer"
- One-line subtitle: short description referencing the δ-mem paper (e.g. "An interactive
  testbed for the Online State of Associative Memory module")
- A small config icon/expander to reveal adjustable parameters (Section 7)

---

## 4. Main Panel

### 4.1 Input Area

- A single text input field, full width of the main panel
- Placeholder text: `"Type a sentence to remember, or a question to ask..."`
- Character counter shown below the field (e.g. `42 / 500`), turns warning color near limit
- Two buttons directly below the input, side by side:
  - **Insert** — writes the current input into memory
  - **Retrieve** — queries memory using the current input
- Buttons are disabled (greyed out) when the input field is empty
- After either action, the input field clears automatically and refocuses for the next entry

### 4.2 Memory Matrix Heatmap

- Positioned directly below the input area — this is the visual centerpiece of the app
- Renders the current memory state matrix `S` as an `r × r` grid (default 8×8)
- Each cell is colored by value magnitude using a single continuous color scale
  (diverging: negative values one hue, positive values another, zero/near-zero neutral)
- A color scale legend is shown alongside the heatmap (min/max value labels)
- **On Insert**: cells that changed most since the last update briefly highlight (subtle
  border pulse or brightness shift, ~400–600ms transition), then settle into their new
  values. This draws the eye to *where* the write affected the matrix.
- **On Retrieve**: the matrix itself does not change (retrieval is read-only). No
  highlight animation plays on the matrix during retrieval — instead, see Section 4.3.
- **Empty state**: before any input, the matrix displays as a flat, near-zero grid with
  a faint placeholder label ("Memory is empty — insert a sentence to begin")
- A small caption beneath the heatmap states the current step count
  (e.g. `Step 6 — 6 sentences written`)

### 4.3 Retrieval Results Area

- Appears below the matrix only after at least one Retrieve action has been performed
- Shows the submitted query text as a small header (e.g. `Query: "Where does Alice live?"`)
- Displays a ranked list of stored sentences with similarity scores:

```
1. Alice lives in Paris.                    score: 0.87
2. Alice's favourite colour is blue.        score: 0.61
3. Bob works at a hospital.                 score: 0.23
```

- The top-ranked result is visually emphasized (bolder text or accent border) since it
  represents the system's actual "answer"
- Scores are shown as a horizontal bar alongside each entry for quick visual comparison,
  not just the raw number
- If the registry is empty when Retrieve is pressed, show a message: "Nothing has been
  stored yet — insert a sentence first."

### 4.4 Reset Control

- A clearly separated "Reset memory" button at the bottom of the main panel
- Requires a confirmation step (e.g. a second click or small confirm prompt) before
  clearing the matrix and registry, to avoid accidental data loss mid-demo

---

## 5. Side Panel — Sentence Registry

- Always visible, scrollable list, most recent entry at the top
- Each entry shows:
  - The original sentence text
  - Its insertion order/step number (e.g. `#4`)
- **Linking behaviour**: clicking a registry entry highlights that entry if it appeared
  in the most recent retrieval results list (Section 4.3), allowing the user to trace
  "this is the fact that got activated by that query." If the entry was not part of the
  last retrieval, clicking it has no special effect beyond a brief selection highlight.
- Registry persists for the full session until Reset is triggered
- A small counter at the top of the panel shows total sentences stored
  (e.g. `6 / 30 stored`)

---

## 6. Visual Style

- Clean, minimal layout — no decorative elements that don't carry information
- One consistent color scale for the matrix; one consistent accent color for
  highlights/emphasis throughout (used for: top retrieval result, active registry
  selection, write-pulse highlight) so the user learns "this color means relevant/active"
- Typography: sans-serif throughout, clear size hierarchy between header, body text, and
  captions/scores
- No unnecessary motion — only the two specified transitions (write-pulse, registry
  selection highlight) are animated; everything else updates instantly

---

## 7. Configurable Parameters Panel

Tucked behind a small expandable section near the header (collapsed by default so it
doesn't distract from the main demo flow):

- Memory size `r` (slider or dropdown, default 8)
- Write strength `β` (slider, default 0.1)
- Max input length (display-only, not user-editable post-launch)
- Max stored sentences (display-only, not user-editable post-launch)

Changing `r` or `β` after sentences have already been inserted triggers the same
confirmation-to-reset flow as Section 4.4, since changing matrix dimensions invalidates
the existing state.

---

## 8. Interaction Summary (User Journey)

1. User sees empty matrix and empty registry on load
2. User types a sentence, clicks Insert → matrix updates with a brief highlight,
   registry gains a new entry, step counter increments
3. User repeats step 2 several times to build up memory
4. User types a question, clicks Retrieve → ranked results appear below the matrix;
   top result is emphasized
5. User can click entries in the registry to see which ones were part of the last
   retrieval
6. User can adjust parameters or reset to start a new memory session

---

## 9. Explicitly Out of Scope for This UI

- No user accounts or saved sessions across visits
- No editing or deleting individual registry entries
- No export of matrix state or results
- No mobile-specific gesture interactions beyond standard responsive stacking