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

## Status: Stage 0, plus Stage 1's catalog and controls

`PLAN.md` is the design spec; **read it before writing anything**, since its
scope boundaries are settled decisions rather than defaults. The sections most
often needed are §3 (dataset/architecture catalog), §7 (technical design), §8
(asset pipeline), and §11 (milestones).

Stage 0 — the numerical and visual spike — is done: the PyTorch export with
golden fixtures, the nnet-svg adapter, editable weights and biases, and the
downstream-effect analysis. The catalog is now complete too: **four datasets
x nineteen variants = 76 checkpoints**, selected by the controls described
below. `README.md` documents the export schema that joins the two halves and
the deviations from `PLAN.md`; keep both current when that contract changes.

**The panels are a canvas the reader arranges**, not a fixed grid
(`view/workspace.ts`, with the geometry in `view/layout.ts`). Each panel is
a card on twelve columns and rows of 16 pixels: drag it by its title bar,
resize it from the corner, fold it down to the bar alone with the chevron.
The point is a figure that can be built up a piece at a time — five views of
one computation all at once is a wall, and a reader who has just met the
network diagram can keep the rest minimised until they are wanted. The
arrangement is the reader's, so it is kept in `localStorage` and
`Reset layout` puts it back.

Two rules do the arranging, and both are pure functions in `view/layout.ts`:
**nothing overlaps** (a panel dropped on another pushes it down, and
whatever that lands on in turn) and **nothing floats** (every panel then
rises until it meets the one above it, so dragging a panel out of the middle
closes the hole). Compaction is vertical only: the column is the reader's
decision and the row is a consequence of it. Which way a displaced panel
goes is decided by its **centre** against the dragged panel's — a panel
dragged *onto* another pushes it down, a panel dragged *past* one lets that
one come up into the room left behind. Without that second case nothing
could ever be reordered downward, because the dragged panel would simply be
floated back where it came from. The arrow keys move a panel too, and
vertically they step past the neighbour rather than by a row, for the same
reason.

Three things about that module are easy to undo by accident. Compaction
places panels **in row order**, with the dragged one winning only its own
row: placing it first instead means a panel resized at the bottom of the
canvas jumps to the top. Heights are **fitted to content once**, after the
first render, because a default height is a guess about a window nobody has
seen and these plots scale with their width; the measurement has to happen
with the panel transitions suppressed, or a box halfway through a 120ms
width transition is measured at the width it is passing through. And a
stored layout is **refused whole** unless it names exactly the panels this
build has, so adding a panel costs every reader their arrangement once —
which is the cheapest honest answer for something that is a convenience.

Panels are **shell-owned**, like the help popups and for the same reason:
a checkpoint switch rebuilds the views, and a panel that went with them
would take the reader's arrangement with it. The inspector therefore
renders into two bodies the shell owns rather than building its own
sections, and hands out the neuron panel's title (which names the selected
unit) instead of printing a heading inside it.

`main.ts` is in two halves, and the split is what makes a reset honest. The
**shell** is built once: the panels, the four `?` popups, the catalog
controls and the display controls. A **session** is built per checkpoint: the
explorer and the five views that read its snapshot. Changing dataset, depth,
width or activation destroys the session and mounts a new one, so parameters
cannot be carried across a change of problem or of shape (PLAN.md section 4)
— the object that held them is gone. Display preferences (the edge mode, the
sample toggle) belong to the reader, so they live in the shell and survive a
switch. So does the reader's *place* in the figure, which is not the same
thing as state the new asset disagrees with: the probe is carried over when
the dataset is unchanged, and the selected unit when the shape is unchanged
(which is exactly the activation-only switch). Both exist so that "change
one thing and look at the same place" works across a switch, and neither
carries a parameter. The new session is mounted only once its checkpoint has arrived, so a
slow fetch leaves the current figure up rather than blanking it. A discrete
slider fires an event for every stop it crosses, so several loads can be in
flight at once: `main.ts` keeps a request token and mounts only the last one
asked for. Nothing is disabled while a load runs — disabling a slider
mid-drag would break the gesture — so supersession, not a busy flag, is what
keeps two switches from fighting.

`view/selector.ts` is the catalog control. Depth and width are **discrete
sliders with labeled stops** (PLAN.md section 3), because the quantities are
ordered; dataset and activation are named rather than ordered, so they are
button groups. A slider's value is a stop *index*, never the quantity, so the
stops need be neither consecutive nor evenly spaced. Deepening gives the new
layer the width of the last one that already existed (2-4-1 becomes 2-4-4-1),
and the width control for a layer that does not exist is **disabled in
place** rather than hidden: the depth slider sits directly above that row,
and a row that reflowed on every depth change would move a control out from
under the pointer that had just used it. A disabled control shows the width
its layer *would* get, so it is a preview of the next stop up rather than a
blank or a stale number, and none of its stops is struck through — the
catalog has those widths, there is just no layer to apply them to yet.
(It was hidden until the layout churn made the case for keeping the row
fixed; do not revert that without solving the reflow.) Every stop is checked against the published catalog before it is
offered: `createCatalogIndex` (`model/catalog.ts`) enumerates the manifest,
unavailable buttons are disabled, unavailable slider stops are struck
through, and one dragged onto anyway snaps back and says why. A depth-0 entry
keys without an activation, so the activation the control still remembers
cannot make the linear model unreachable.

The "Inputs → output" panel is one figure on shared axes: the ŷ(x₁, x₂) map,
ŷ(x₁) below it holding x₂ at the probe, and ŷ(x₂) to its left holding x₁,
rotated so x₂ still runs upward and ŷ grows away from the map. One pair of
crosshairs crosses all three and each slice's curve meets it at the
prediction, so a misaligned axis shows up immediately; `tests/view.dom.test.ts`
pins the alignment. The map also draws its own two axes — thin dashed lines
where each input is zero, under the samples and the crosshairs — so a
position can be read without counting in from the edges. Training and held-out samples are drawn **alike**: the split is a number in
"Prediction and loss", not a glyph to decode. No checkpoint in this catalog
can show a gap the map would explain — the two regression generators are
noiseless, so their splits agree by construction, and the classifiers'
irreducible error is in the overlap of the classes rather than in which
split a point landed in. Do not reintroduce a dashed outline for the
held-out split without a dataset whose splits actually differ.

While the samples are visible a drag that lands inside a dot **snaps onto
that sample**: the probe becomes the sample rather than a free point beside
it, which is the only way to read a real target and a real point loss out
of this panel, and on a classifier the only way at all, since an arbitrary
point in a stochastic dataset has no label. The snapped dot is drawn larger
and last, so it is over the crosshairs rather than under them. Snapping is
gated on the samples actually being drawn: taking the probe onto something
invisible would silently change what the probe *is* — a sample probe has a
target where a free one may not — with nothing on screen to explain it.

It opens with no
samples drawn and the probe at `OPENING_PROBE` (`state.ts`), 0.5 in each
input, clamped into that input's domain: the panel is there to be probed
first, `Show samples` is an explicit opt-in, and the probe starts off the
axes it would otherwise hide, with no input term of the first layer's sums
zeroed.

The display modes now only switch what the **edges** show (weight or
contribution). Node fills never vary with the mode.

The output node is the prediction — its fill and its badge — so **the
output scale stands up beside it**, in a gutter down the right of the
frame, with the prediction marked by a triangle on its left and the probed
sample's target by one on its right, under a rotated "target" label. The
bar is the axis that node's own fill came off, so the two numbers the loss
is made of are a pair of marks on one scale rather than two printed numbers
the reader has to relate; it replaced a printed target for that reason. The
gutter is reserved whether or not the bar is drawn — the network is laid
out in `NETWORK_BOX`, 56 units narrower than the frame — because a frame
that grew when a sample was selected would resize the whole diagram at the
moment the reader was looking at one node of it.

It is drawn only when the samples are shown **and** the probe is on one of
them: both conditions the reader can see on screen. A free probe has a
target too on a generator with an analytic one, but marking it would put a
number on the bar with nothing on screen to say where it came from. A
classifier's target is a class label rather than a point on the logit axis
the bar spans, so its marker goes to the end of the scale its class wants,
exactly as the map draws that same sample (PLAN.md section 3); a prediction
that runs past the scale is pinned to the end it ran past, with the real
number in the marker's title and on the node's badge. The strip is drawn as
a stack of bands rather than as an SVG gradient, so each band is a
`color-mix` of the theme's own tokens and a theme change repaints it with
no re-render.

**A classifier's output is a logit, and the panels say so.** Two of the four
datasets are classifiers, so this is not a hypothetical: sigmoid is applied
for display and labeled, and the logit stays beside it, because the logit is
what the parameters produce and what every plot draws (PLAN.md section 3).
The inspector reads `logit z`, `probability p` and `class y` rather than
`prediction ŷ` and `target y`, and its sensitivity defaults to `∂p/∂x` with
`∂z/∂x` kept and labeled. The output colour scale stays the diverging one
centred on zero, which is exactly right here: z = 0 is p = 0.5, so the
neutral band in the map *is* the decision boundary. The one thing that
cannot stay on the logit axis is the target — 0 and 1 are class labels, not
logits — so a classifier's targets are left out of the scale's limit and a
sample is drawn at the end of the scale its class wants. A sample over a
neutral background is then one the model is unsure about. An explicit dashed
p = 0.5 contour (PLAN.md section 4) is still unimplemented; the neutral band
stands in for it.

"Parameter Tensors" (`view/tensors.ts`) is the parameters by shape rather
than by value: one group per parameter layer, left to right, each the bias
vector drawn above its weight matrix, with one light dot per slot and no
numbers. A **column** of a group is one destination unit — its bias and
every weight feeding it — so the selected unit shades as a single vertical
band across both rectangles, which is the same thing the network panel's
selection means. Rows of a weight matrix are the sources. (The columns run
this way round because the bias strip sits above the matrix and has to line
up with it.) Each rectangle carries an SVG `<title>`, so hovering **anywhere inside it**
names it: "2×4 weights connecting the input layer to hidden layer 1". The
title is the group's first child and `.tensor-frame` sets
`pointer-events: all`, because the frame is unfilled and an unfilled shape
is not hit-tested — without that the tooltip answers only over the slots,
and the gaps between them, which are most of the rectangle, say nothing.
Every group shares the same two bands, so they are named once in a rotated
gutter down the left — "bias" above, "weight" below — rather than by a pair
of labels repeated per layer.

Each of the four plot panels keeps its explanation in a popup behind a `?`
in its heading rather than printed underneath it (`view/help.ts`). The popup
is a child of `document.body` and is placed **beside** its panel — right if
there is room, else left, else below — so it never covers the thing it
explains. `createHelp` *moves* the legend element into the popup instead of
copying its text, so whatever writes that text (`main.ts` for two panels,
`paramspace.ts` for its own) keeps the popup current; `main.ts` therefore
holds those elements by reference, since they are no longer reachable
through `#app`.

**Selection is a *unit*, not a single parameter, and is never empty.**
Clicking a node selects that neuron and every edge feeding it; clicking an
edge selects the neuron it *feeds*, with its siblings into the same neuron.
The explorer opens on the output unit and its final-layer edges. An input
node owns no parameters, so clicking one selects nothing and leaves the
current unit alone: there is no way to deselect. `UnitRef` and
`unitParameters` (`model/types.ts`) are the whole vocabulary; the latter
orders a unit's parameters the way the UI lists them — the bias first, then
the weights by source index, which is top to bottom as nnet-svg draws them.

"Parameters → output" superimposes one curve per parameter **of the selected
unit** — its bias and the weights feeding it, nothing else: the output as
that parameter alone is swept across the parameter scale, everything else
held. Every curve passes through the current output at the current value, so
**every dot lands on the horizontal output line** — the quickest check that
the panel is right, and what `tests/view.dom.test.ts` asserts. Each curve is
named at its dot (`b`, `w₀`, …) and has one slider under the plot in the
same order. The parameter colour bar is that panel's x axis, its background
is the output scale stood up (so a curve's height and the colour behind it
are the same number), and the plot, the bar and every slider share one grid
column so all of them are the same axis. Those sliders are now the only way
to edit a parameter: the inspector's parameter table and its per-parameter
restore button are **removed**, so values beyond the ends of that axis are
unreachable until something replaces them.

The inspector is two panels: "Prediction and loss", which also carries the
predicted-vs-actual comparison after an edit, and "Neuron: …", which reads
top to bottom as the neuron computes: one table row per input term (`i=n`,
`wᵢ`, `aᵢ`, `wᵢ·aᵢ` — no bias row, since the bias is not a term of the sum),
then `Preactivation z = Σ wᵢ aᵢ + bias = …` and `Activation a = ReLU(z) = …`
written out with their values, and beside all of that — to the right while
the panel is wide enough, wrapping underneath when it is not — **a plot of
the activation function with that unit's own (z, a) marked on it**. Nothing
follows it. The slope `φ′(z)` stays among those lines because it is the only
place the ReLU convention is disclosed (PLAN.md section 4). Both panels'
numbers sit next to their labels rather than at the far edge: the readouts
are a two-column grid of `max-content` (the rows are `display: contents`)
and the contributions table is `width: auto`. The inspector row is a wrapping
flex row of content-width panels, not equal columns: the neuron panel needs
the room its plot takes and the prediction readouts do not.

The line under the panels is standing instructions only. It used to
announce each edit (`W[0][2][0]: -0.642 → -0.645`); an edit's before and
after are already in the slider's readout and in the predicted-vs-actual
comparison, so nothing below the fold changes as the explorer is used and
nothing down there can go stale. The plot is
square with one scale on both axes, so the identity and ReLU's on-arm are
both at 45° and what the frame shows is the function's own shape; the output
layer applies no nonlinearity, so its unit plots the identity. z and a come
from the trace, so moving the bias slider — or any weight of that unit —
slides the dot along the curve, which itself never moves. The axis range is
the smallest round number holding that z with 5% headroom, so the dot cannot
land on the frame and the range only steps when z really outgrows it.
`tests/view.dom.test.ts` asserts the dot is a point *of* the drawn curve.

What is left of Stage 1 is the dataset-specific display work the catalog now
makes reachable: the p = 0.5 decision contour, and whether anything else
should read differently for a classifier.

## Commands

```bash
uv sync                  # create .venv from uv.lock (Python 3.12)
uv run nnintro train     # curate + train the catalog -> data/
uv run nnintro publish   # data/ -> frontend/public/data (the tracked snapshot)
uv run nnintro verify    # re-check the published snapshot
uv run pytest            # 26 pipeline tests, ~1 s
                         # `nnintro train` runs 76 configurations x 256
                         # attempts over a process pool: ~40 minutes, and it
                         # exits non-zero if any configuration ends up with
                         # no checkpoint. `--only <substring>` restricts it.

cd frontend
npm ci                   # npm install needs --legacy-peer-deps on a fresh
                         # resolve; see README
npm run test             # vitest, including the golden fixtures for every
                         # one of the 76 checkpoints
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
- `model/ripple.ts` decides what an edit actually changed. Nothing in the UI
  displays it any more: its animated counterpart (`view/ripple.ts`, zipline)
  and then the inspector's "Downstream effect" table were both removed at
  the user's request. The module and `tests/ripple.test.ts` are kept, as is
  `@dkirkby/zipline` in `package.json`, against the animation coming back.
- `model/surface.ts` evaluates the prediction surface over the input domain,
  and the one-input slices through the probe. `main.ts` recomputes the
  surface only when the parameters change; the slices follow the probe and
  are recomputed every render.
- `model/curves.ts` sweeps one parameter at a time with everything else
  held, for the "Parameters → output" panel. The caller names the parameters
  to sweep (the selected unit's). It mutates a private copy: the explorer's
  own parameters must come back untouched.
- `state.ts` owns the baseline/working split, the selected unit, history, and
  the one versioned snapshot every view consumes.
- `view/adapter.ts` is the **only** place model and renderer indices meet.
- `view/colormap.ts` owns both colour maps. `main.ts` builds one
  `Palette` per render and hands the same object to both panels.
- `view/network.ts`, `view/inputoutput.ts`, `view/paramspace.ts`,
  `view/inspector.ts` render a snapshot and own no mathematics.
- `view/layout.ts` is the workspace's geometry and has no DOM at all, the
  same split as `model/` against `view/`: what pushes what is arithmetic,
  and `view/workspace.ts` turns it into elements and gestures.

### Invariants that are easy to break

- **Training alone does not produce a teaching asset.** Plain training on the
  tilted plane reliably routes everything through one hidden unit and leaves
  the rest with ~1e-6 output weights: it fits perfectly and demonstrates
  nothing. `train.curate` trains 256 times and keeps the lowest training loss
  among the attempts that pass `SELECTION_CRITERIA`, and the report is
  exported under `selection` so the choice can be audited. Do not "simplify"
  this back to a single `train()` call, and do not lower the attempt count.
  The hardest entry is the tilted plane at 2-4-4-1 with ReLU: eight hidden
  units fitting a two-parameter plane leaves most of them with nothing to
  do, so only about **one attempt in fifty** is acceptable and its first
  acceptable seed is the **114th** — 96 attempts leaves that configuration a
  hole, which is how the budget came to be 256 rather than 96. The extra is
  not rounding: at 128 that entry would rest on two seeds, and the
  acceptance pattern moves with numpy and torch.
  - The criteria are about **legibility, not accuracy**, and they have to be:
    a linear model on XOR and a 2-2-1 on the curved surface cannot fit their
    datasets, and those imperfect fits are intended assets (PLAN.md section
    3). An absolute loss threshold was the Stage 0 criterion and only ever
    worked because there was one dataset. The checks are: no vestigial unit
    (`relative_output_weight`), no dead or saturated unit
    (`relative_variation` — the std of the unit's activation over a domain
    grid, which catches a dead ReLU, a saturated tanh and a ~0-weight unit
    alike), and at least one hidden unit whose transition crosses the domain
    (`positive_fraction` strictly inside (0, 1), which for ReLU is exactly
    the kink crossing). Every hidden layer is diagnosed, not just the first.
  - The objective is the **training** loss, deliberately not the held-out
    loss: the explorer shows both, and picking the attempt with the best
    held-out loss would bias that number below its training counterpart and
    make the gap an artefact of selection.
- **Two colour maps, one per family of quantity.** `view/colormap.ts` owns
  them: **param** (magenta/green) for weights and biases — every edge, and
  the hidden nodes, which are filled by their bias while their number stays
  the activation; **output** (red/blue) for the prediction — the surface, the
  slice backgrounds, the samples, the output node and the background of the
  parameters-to-output plot. A single map across every quantity made each
  panel look like the others and invited comparisons between unrelated
  numbers, so do not re-merge them. The inputs had a third map and no longer
  do: their axes carry their numbers, so input nodes are left unfilled and
  the two axes are plain label strips. Do not add it back without a use.
  Within a family a colour means one number: the output node is the colour of
  the surface under the probe. Each limit is a function of the parameters and
  the dataset alone, never of the probe, or dragging the probe would recolour
  the panel under it. The **output** limit is narrower than that, and
  `outputLimit` (`view/colormap.ts`) is the only place it is decided: for a
  regression it is `dataset.targetRange`, the range of the targets and
  nothing else, so it does not move when the parameters do and the colour
  under the probe means the same number either side of an edit. That range
  is the **generator's**, declared in `datasets.py` and published with every
  checkpoint, not the drawn samples': both generators top out at |y| = 1 by
  construction — the tilted plane's coefficients were chosen for it (see
  README's deviations) — but 400 uniform draws do not land on the corner
  where the plane peaks, so a scale fitted to the sample would end at 0.91
  instead of 1 and would move with the sampling seed. The pipeline is the
  only thing that can know the range (it has the formula *and* can search
  the domain), which is why this is an asset field rather than a browser
  computation. Where a generator declares no range, the limit falls back to
  `snugLimit(dataset.samples.y)`, the drawn targets being the only range
  there is; `snugLimit` rather than `niceLimit` because a limit that cannot
  shift underfoot does not need a round ladder to hold it still, and the
  ladder would spend most of the scale — the tilted plane's targets used to
  reach 1.07, which the ladder rounded to 1.5, leaving even its most
  extreme sample at 71% of its colour and nothing on that map ever fully
  red or blue. The consequence is deliberate — a prediction outside the
  targets' range saturates, and `Randomize all` paints the whole map at one
  end of the scale, which is the honest reading of a model nowhere near its
  data. Do not put `surface.extent` back into it. A classifier has no such
  range (0 and 1 are class labels, not logits, and the surface is a logit),
  so there the limit *is* the surface's extent and its samples are drawn at
  the ends of that scale rather than at values on it. Zero is the neutral surface colour: do not reintroduce a
  brightness floor to make small values visible, because it moves zero off
  neutral. Edge widths carry that floor instead.
- **An edge the scale cannot show still has to be visible and selectable.**
  Below 5% of the scale an edge is drawn as a dotted grey line (`.is-faint`,
  a CSS rule, which outranks the renderer's `stroke` presentation attribute),
  and every edge carries a wide transparent twin with the same `data-key` for
  clicking. The twins go between nnet-svg's `.dn-edges` and `.dn-items`
  groups, or they would swallow clicks meant for a node. An edge's width
  ramps from 2.2 to 7 user units over the scale; the floor is that high
  because the view box (680 units) is wider than the panel that renders it,
  so a 1-unit stroke lands under a pixel and reads as nothing at all.
- **Selection is black, never the accent.** The accent green is a parameter
  colour now. The selected unit's node takes a thick ink outline of its own;
  each edge feeding it keeps its own colour *and* its own thickness and is
  railed by two thin ink lines, one along each side, offset along the edge's
  normal in a group inserted before `.dn-edges`. So a selection is one node
  plus one rail pair per incoming edge, never a lone edge.
- **The parameters-to-output axis is fitted to the selected unit's sweeps,
  with 5% headroom.** It is the one axis in the app that is *not* a fixed
  scale, because a dead ReLU's sweeps move the output by almost nothing and
  a shared axis would flatten them to a line. The headroom exists for the
  case where the current output is itself the extreme of every sweep —
  common when the unit is off — which would otherwise draw the output line
  along the frame. `view/colormap.ts`'s `niceLimit` snaps its result back to
  12 significant digits because that number is printed as a tick. The
  neuron panel's activation plot is fitted the same way, for the same
  reason: its dot has to stay inside the frame as an edit moves it.
- **The slices' value axis is the output scale, not their own range.** That
  is what makes a slice's height and its background two readings of one
  number, and its ticks are the only numbers the output scale has. Fitting
  the axis to the curve would read better in isolation and break the panel.
- **`state.ts` records an edit BEFORE it notifies subscribers.** The views
  build the predicted-vs-actual comparison from `lastEdit()` inside the
  subscription; assigning after `commit()` hands them the previous edit, and
  the comparison silently vanishes. `tests/state.test.ts` pins this.

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
