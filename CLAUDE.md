# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An interactive browser explorer for **dense neural networks** — inputs, weights, biases,
activations, predictions, loss, and derivatives — for students with a science background.
Part of UCI's AI4SCI Deep Learning course; `../infoplane`, `../mutual`, `../ntk`,
`../polyfit` are sibling course projects and the closest models for how this one should be
built.

The central interaction is **change one thing, predict what will happen, inspect the
consequences**: load random or pre-trained parameters, trace one input through the network,
edit a single weight or bias, and compare the local derivative's prediction against the
actual change.

## Status: Stage 0 is implemented

`PLAN.md` is the design spec; **read it before writing anything**, since its
scope boundaries are settled decisions rather than defaults. The sections most
often needed are §3 (dataset/architecture catalog), §7 (technical design), §8
(asset pipeline), and §11 (milestones).

Stage 0 — the numerical and visual spike — is done: one dataset, one network,
the PyTorch export with golden fixtures, the nnet-svg adapter, editable
weights and biases, and the downstream pulse. `README.md` documents the export
schema that joins the two halves and the deviations from `PLAN.md`; keep both
current when that contract changes.

The "Input → output" panel is one figure on shared axes: the ŷ(x₁, x₂) map,
ŷ(x₁) below it holding x₂ at the probe, and ŷ(x₂) to its left holding x₁,
rotated so x₂ still runs upward and ŷ grows away from the map. One pair of
crosshairs crosses all three and each slice's curve meets it at the
prediction, so a misaligned axis shows up immediately; `tests/view.dom.test.ts`
pins the alignment. It opens with no samples drawn and the probe at the centre
of the domain: the panel is there to be probed first, and `Show samples` is an
explicit opt-in.

The display modes now only switch what the **edges** show (weight or
contribution). Node fills never vary with the mode.

"Parameters → output" superimposes one curve per parameter: the output as
that parameter alone is swept across the parameter scale, everything else
held. Every curve passes through the current output at the current value, so
**every dot lands on the horizontal output line** — the quickest check that
the panel is right, and what `tests/view.dom.test.ts` asserts. The parameter
colour bar is that panel's x axis, and the editing slider sits under it on
the same axis; the inspector keeps the typed value and the restore button
for values beyond the axis.

Stage 1 is mostly catalog growth: `CATALOG` in `src/nnintro/cli.py` has one
entry and needs the other 75, and the UI needs the depth/width/activation
controls that select among them.

## Commands

```bash
uv sync                  # create .venv from uv.lock (Python 3.12)
uv run nnintro train     # curate + train the catalog -> data/
uv run nnintro publish   # data/ -> frontend/public/data (the tracked snapshot)
uv run nnintro verify    # re-check the published snapshot
uv run pytest            # 26 pipeline tests, ~1 s

cd frontend
npm ci                   # npm install needs --legacy-peer-deps on a fresh
                         # resolve; see README
npm run test             # 82 vitest tests, including the golden fixtures
npm run typecheck
npm run build            # static site -> frontend/dist
```

No linter or formatter is configured, matching the siblings.

## Expected stack and conventions (from the sibling projects)

`PLAN.md` §7 proposes TypeScript + Vite + a thin React layer, with pure numerical modules
independent of the UI, plus an offline PyTorch pipeline. The siblings establish how that is
normally done here:

- **Frontend** (`../mutual`): Vite + TypeScript + vitest, `npm run dev` / `build` / `test`,
  **exact pinned versions** in `package.json` (no `^`), d3 for plots, deployed to GitHub
  Pages by a `.github/workflows/` job on push to `main`.
- **Python** (`../infoplane`, `../ntk`): `uv` with `pyproject.toml` + `uv.lock`, Python
  3.12, `uv sync` / `uv run pytest` / `uv run <console-script>`, pytest only — no linter or
  formatter is configured in any sibling.
- **Committed asset snapshots**: `../infoplane` keeps gitignored working output in `data/`
  and a separate *tracked* snapshot under `frontend/public/`, written only by an explicit
  `publish` command. The Pages build runs no Python, so a push deploys the snapshot. Expect
  the same split here for checkpoints, and the same rule: never hand-edit the snapshot.
- **Torch seeding** uses `torch.random.fork_rng` so constructing a model never perturbs the
  caller's RNG (`../ntk`, `../infoplane` both do this).

`nnet-svg` and `zipline` are the user's own packages (`zipline` is also checked out at
`~/UCI/AI4SCI/code/MLP/zipline`). Verify their APIs against pinned commits during the
Stage 0 spike; their READMEs on `main` are mutable and are what `PLAN.md` was written from.

## Architecture

Two halves joined by the JSON contract in `README.md`.

**Pipeline** (`src/nnintro/`), a straight line, each module independently
testable: `datasets.py` (fixed seeded samples) → `model.py` (`DenseNet`,
losses) → `train.py` (train one config, then **curate**) → `export.py` (the
asset contract and the golden probes) → `cli.py` (`train`/`publish`/`verify`).

**Browser** (`frontend/src/`), with the numerics kept free of the DOM:

- `model/evaluate.ts` is the source of mathematical truth — forward pass,
  input sensitivity, point and mean-loss gradients. No DOM import belongs here.
- `model/ripple.ts` decides what an edit actually changed. Its animated
  counterpart (`view/ripple.ts`, zipline) is **removed for now** at the
  user's request; the inspector's "Downstream effect" table still shows the
  reachable-vs-changed distinction, and `@dkirkby/zipline` stays in
  `package.json` against the animation coming back.
- `model/surface.ts` evaluates the prediction surface over the input domain,
  and the one-input slices through the probe. `main.ts` recomputes the
  surface only when the parameters change; the slices follow the probe and
  are recomputed every render.
- `model/curves.ts` sweeps one parameter at a time with everything else
  held, for the "Parameters → output" panel. It mutates a private copy: the
  explorer's own parameters must come back untouched.
- `state.ts` owns the baseline/working split, selection, history, and the one
  versioned snapshot every view consumes.
- `view/adapter.ts` is the **only** place model and renderer indices meet.
- `view/colormap.ts` owns all three colour maps. `main.ts` builds one
  `Palette` per render and hands the same object to both panels.
- `view/network.ts`, `view/inputoutput.ts`, `view/paramspace.ts`,
  `view/inspector.ts` render a snapshot and own no mathematics.

### Two invariants that are easy to break

- **Training alone does not produce a teaching asset.** Plain training on the
  tilted plane reliably routes everything through one hidden unit and leaves
  the rest with ~1e-6 output weights: it fits perfectly and demonstrates
  nothing. `train.curate` trains repeatedly and keeps the best checkpoint that
  passes `SELECTION_CRITERIA` (no vestigial units, no dead units, at least one
  ReLU kink crossing the visible domain), and the report is exported under
  `selection` so the choice can be audited. Only 2 of 24 attempts pass, so do
  not "simplify" this back to a single `train()` call.
- **Three colour maps, one per family of quantity.** `view/colormap.ts` owns
  them: **input** (brown/teal) for the two coordinates — the input nodes and
  the two axis bars; **param** (magenta/green) for weights and biases — every
  edge, and the hidden nodes, which are filled by their bias while their
  number stays the activation; **output** (red/blue) for the prediction — the
  surface, the slice backgrounds, the samples and the output node. A single
  map across all three made every panel look alike and invited comparisons
  between unrelated numbers, so do not re-merge them. Within a family a
  colour still means one number: the output node is the colour of the surface
  under the probe, and an input node is the colour its axis bar shows at that
  coordinate. Each limit is a function of the parameters and the dataset
  alone, never of the probe, or dragging the probe would recolour the panel
  under it. Zero is the neutral surface colour: do not reintroduce a
  brightness floor to make small values visible, because it moves zero off
  neutral. Edge widths carry that floor instead.
- **The slices' value axis is the output scale, not their own range.** That
  is what makes a slice's height and its background two readings of one
  number, and its ticks are the only numbers the output scale has. Fitting
  the axis to the curve would read better in isolation and break the panel.
- **`state.ts` records an edit BEFORE it notifies subscribers.** The views
  build the ripple and the predicted-vs-actual comparison from `lastEdit()`
  inside the subscription; assigning after `commit()` hands them the previous
  edit, and both silently vanish. `tests/state.test.ts` pins this.

## Design decisions that are load-bearing

These come from `PLAN.md` and are the ones most likely to be "fixed" back into bugs:

- **Edge indexing is destination-first.** Canonical model storage is `W` with shape
  `[out, in]`, matching PyTorch `Linear`. The current nnet-svg examples index edge values
  **source-first**. Centralize that translation in one adapter and test it with an
  asymmetric matrix.
- **No training in the browser.** All training and checkpoint generation happen offline in
  PyTorch. No optimizer controls, no training UI, no arbitrary model import.
- **No backpropagation lesson.** Derivatives are computed analytically and presented as
  observable properties of the current model. No backward-flow animation, no chain-rule
  traversal, no optimizer steps — a separate tool covers that topic.
- **Every architecture/activation combination a control can reach must exist in the
  catalog** (4 datasets × 19 variants = 76 checkpoints). Use discrete sliders with labeled
  stops; a reachable combination without a checkpoint is a bug. An imperfect fit (linear
  XOR) is an intended teaching asset, not a failure.
- **Immutable baseline + editable working copy.** Changing dataset, architecture, or
  activation creates a fresh baseline and announces the reset; never silently resize
  matrices or reuse parameters across problems. Provenance is always displayed ("Random",
  "Pre-trained", "Modified from …") with seed or checkpoint ID. Randomize-all records the
  actual parameters, not just the seed.
- **Differentiate the mean loss; never average per-example gradient magnitudes.** Point
  loss is `ℓ = (ŷ − y)²` (regression) or BCE-from-logits (classification); dataset loss `L`
  is the arithmetic mean over the chosen split. Label which objective any gradient overlay
  uses.
- **Classification keeps logit and probability distinct**, uses a numerically stable
  BCE-from-logits, and defaults sensitivity to probability with logit available and labeled.
- **ReLU's kink**: disclose the convention (derivative 0 at z = 0) and that no unique
  derivative exists there. Mark non-finite states rather than clamping them away.
- **The ripple animation is never the source of truth** (the animation itself is removed
  for now; this still governs if it returns). Compute the new state immediately,
  animate the explanation separately, preserve before/after values. Distinguish
  structurally reachable nodes from nodes whose values actually changed — ReLU can block a
  change and contributions can cancel. Zipline's carrier oscillation, speed, and amplitude
  carry **no** quantitative meaning; deltas go in labels and documented color scales. New
  edits cancel obsolete animations; replacing a network destroys its controllers.
- **Don't overload one color scale** with activations, weights, contributions, and
  sensitivities at once — they are explicit display modes, each with numeric values, signs,
  legends, and a parameter table as non-color, non-SVG-picking alternatives.
- **Probe rules**: show point loss only where a target is actually defined. Never assign a
  label to an arbitrary point in a stochastic classification dataset.

## Verification expectations

`PLAN.md` §12 is the acceptance standard, and §8 requires exported **golden probes**
(inputs, all preactivations/activations, predictions, losses, input sensitivities, parameter
gradients) alongside each checkpoint. Browser forward passes and analytic derivatives are
checked against those PyTorch fixtures for every activation and task, with centered finite
differences as an independent check (scale-aware steps, away from ReLU kinks). Saved
gradients are validation fixtures only — the browser always recomputes after an edit.
