# nnintro — dense neural network explorer

An interactive explorer for students with a science background: load a trained
or random model, follow one input through the network, edit a single weight or
bias, and compare what the local derivative predicted against what actually
changed.

[`PLAN.md`](PLAN.md) is the design spec. **Stage 0 (the numerical and visual
spike) is implemented**, and so is the catalog growth that Stage 1 is mostly
made of: all four datasets, all seven architectures, all three hidden
activations — 76 checkpoints — with discrete controls that select among
them. Stage 2 completes the initial release.

## Layout

```
src/nnintro/      offline PyTorch pipeline (datasets, model, training, export)
tests/            pytest, for the pipeline
frontend/         the browser explorer: Vite + TypeScript, no framework
frontend/tests/   vitest, including the golden-fixture comparison
```

## Commands

```sh
uv sync                              # create .venv from uv.lock (Python 3.12)
uv run nnintro train                 # train the catalog -> data/
uv run nnintro publish               # data/ -> frontend/public/data (tracked snapshot)
uv run nnintro verify                # re-check the published snapshot:
                                     # hashes, golden probes, split losses,
                                     # and that the catalog is fully covered
uv run nnintro train --only 2-4-1    # retrain a subset, by id substring
uv run pytest                        # pipeline tests
uv run pytest tests/test_export.py::test_golden_probes_reproduce_from_parameters

cd frontend
npm ci                               # from the committed lockfile
npm run dev                          # the explorer
npm run test                         # vitest, including the golden fixtures
npm run typecheck
npm run build                        # static site -> frontend/dist
```

`npm ci` installs from the committed lockfile. A fresh `npm install` (no
lockfile) currently trips an npm/arborist bug while resolving vitest's optional
peer set — `Cannot read properties of null (reading 'edgesOut')` — and needs
`npm install --legacy-peer-deps` once; the resulting lockfile then installs
cleanly with plain `npm ci`.

Note that the published npm package `zipline` is **not** this project's
dependency: the one it uses is `@dkirkby/zipline`. The pulse animation it
drove is removed for now, so the dependency is currently unused and kept
only against its return.

No linter or formatter is configured, matching the sibling course projects.
`data/` is gitignored working output; `frontend/public/data/` is the tracked
snapshot the site is built from, and `nnintro publish` is the only thing that
should write it.

## The export contract

`uv run nnintro train` writes one JSON file per checkpoint plus a
`manifest.json`. The browser enumerates the manifest rather than assuming the
catalog is covered: a control must never be able to reach a configuration that
has no checkpoint. The manifest carries a `datasets` array as well as its
`checkpoints`, so the dataset control can be labelled and grouped without
opening 19 checkpoints to find out what they share.

A checkpoint is **self-contained**, which means each one carries its own copy
of its dataset's 400 samples: the four datasets are therefore stored 19 times
each, and the tracked snapshot is about 3.6 MB. Only one checkpoint is fetched
at a time, so this costs page weight rather than load time. Factoring the
samples into four dataset files would cut the snapshot to roughly a quarter of
that, at the price of a second fetch and a checkpoint that can no longer be
verified on its own.

A checkpoint holds:

| Field | Contents |
| --- | --- |
| `schemaVersion`, `checkpointId`, `contentHash` | provenance; the hash is SHA-256 over the canonical JSON with the hash field removed |
| `dataset` | id, short label, task, domain, input/target names, generator version, the generator's own range of targets over the domain, and the fixed seeded samples with their split membership |
| `architecture` | layer sizes, hidden activation, output transform, loss name, weight layout, and the ReLU derivative convention |
| `parameters` | per layer, `W` with shape `[out][in]` and `b` with shape `[out]` |
| `training` | the full `TrainConfig`, torch version, dtype, final losses |
| `splitGradients` | per split: count, mean loss, and the gradient **of that mean loss** |
| `goldenProbes` | fully worked examples: every preactivation and activation, the output, the point loss, the input sensitivities, and the point-loss parameter gradient |

### Conventions that the schema pins down

- **`W` is `[out][in]`** — destination-first, matching `torch.nn.Linear`.
  nnet-svg indexes edge values source-first; `frontend/src/view/adapter.ts` is
  the single place that translates, and `frontend/tests/adapter.test.ts` pins
  it with an asymmetric matrix.
- **Loss has no hidden factor of two.** Regression point loss is
  `(yhat - y)^2`; the dataset loss is the arithmetic mean over the chosen
  split. Classification uses BCE from logits with the same averaging.
- **`splitGradients` differentiates the mean loss.** It is not a mean of
  per-example gradient magnitudes, which is a different quantity.
- **`dataset.targetRange` is the generator's range, not the sample's.** It is
  what the browser's output colour scale spans, and the two differ: the
  tilted plane's targets reach 1 at the corner (+1, -1), and 400 uniform
  draws do not land on a corner, so a scale fitted to the draw would end at
  0.91 and would move with the sampling seed. The range over a domain is a
  property of the formula, so `datasets.py` declares it and
  `tests/test_datasets.py` checks each declaration against a dense grid. It
  is `null` for a classifier, whose targets are class labels rather than
  values on the logit axis its panels draw; the scale is then the surface's
  own extent.
- **`reluDerivativeAtZero` is 0**, stated in the asset rather than assumed by
  the reader. The `relu-kink` golden probe sits exactly on a kink.
- **Everything is float64**, on both sides. The browser recomputes in
  JavaScript doubles, and the fixtures agree to ~1e-12; training in float32
  would put a 1e-7 floor under a check that should be much sharper.
- **Golden gradients are fixtures, not a runtime cache.** A parameter edit
  invalidates them immediately, which is the entire point of the explorer, so
  the browser always recomputes.

## Golden probes

Each checkpoint ships a probe set chosen for the cases that break
implementations, not for coverage of the input space:

| Probe | Why it is there |
| --- | --- |
| `train-sample`, `test-sample` | labeled examples, so point loss and its gradient are defined |
| `origin` | every preactivation reduces to its bias, so a transposed `W` is invisible here and obvious elsewhere |
| `free-probe` | asymmetric coordinates — the case that catches a transposed `W` |
| `most-inactive` | the grid point with the most inactive ReLU units, where a downstream ripple is **blocked** rather than propagated |
| `relu-kink` | exactly on a kink, found by solving `w . x + b = 0` rather than by search, pinning `phi'(0) = 0` |

`frontend/tests/golden.test.ts` replays all of them through the browser
evaluator and asserts agreement with PyTorch to 1e-9, and
`frontend/tests/derivatives.test.ts` independently checks both kinds of
derivative against centered finite differences.

## Deviations from `PLAN.md`

- Training is followed by a **curation** step the plan calls for in section 8
  but does not specify: `train.curate` runs 256 attempts and keeps the one
  with the lowest training loss among those that pass `SELECTION_CRITERIA`.
  Without it, training reliably produces a model that fits the plane
  perfectly through a single hidden unit and teaches nothing. The chosen
  model's per-unit diagnostics ship under `selection`.
  - The criteria are about **legibility, not accuracy**: no vestigial units,
    no dead or saturated units, and at least one hidden unit whose transition
    crosses the visible domain. Accuracy cannot be a criterion across this
    catalog, because the configurations that *cannot* fit their dataset — a
    linear model on XOR, a 2-2-1 on the curved surface — are intended
    teaching assets (PLAN.md section 3). The report records the best training
    loss any attempt reached alongside the best among those worth showing, so
    what legibility cost is auditable.
  - The objective is the **training** loss, deliberately not the held-out
    loss. The explorer displays both, and selecting on the held-out number
    would bias it below its training counterpart, making the one quantity
    that is supposed to say something about generalization an artefact of
    selection instead.
  - 256 attempts is what the catalog needs to come out complete. The hardest
    entry is the tilted plane at 2-4-4-1 with ReLU: eight hidden units
    fitting a two-parameter plane leaves most of them with nothing to do, so
    only about one attempt in fifty survives the unit criteria and its first
    acceptable seed is the 114th. The budget was 96 until that configuration
    became a gap; the rest above 114 is room for the acceptance pattern to
    shift under a different numpy or torch, since at 128 it would rest on
    two seeds. A configuration that still cannot be curated is reported as a
    GAP and makes `train` exit non-zero, rather than silently leaving a hole.
- **The tilted plane is `y = 0.5 x1 - 0.3 x2 + 0.2`**, not the
  `0.6 x1 - 0.4 x2 + 0.2` of PLAN.md section 3's table. The coefficients are
  equally interpretable — that is the dataset's whole purpose — but these
  sum so that |y| reaches exactly 1 at the corner (+1, -1) and nowhere
  exceeds it. The browser's output colour scale spans a dataset's declared
  `targetRange`, so full red and full blue now mean "the most extreme target
  there is" on a scale whose ends are round numbers. Keep
  `|a| + |b| + |c| = 1` if these are ever changed again;
  `tests/test_datasets.py` pins it.
  `GENERATOR_VERSION` went to 2 at the same time, so a checkpoint built
  against the old formula is identifiable rather than silently mixed in.
- **The panels are a canvas the reader arranges**, where PLAN.md section 4
  asks for "three coordinated areas". The three areas are still the default
  arrangement, but each panel can be folded down to its title bar, dragged
  anywhere on a twelve-column grid, and resized within what its content can
  live with; the layout is kept between sessions and `Reset layout` restores
  the default. The point is that the figure can be built up a piece at a
  time rather than arriving as five views at once. The plan's requirement
  for narrow screens is unchanged: below 860px the panels stack and the
  stored arrangement is simply not applied.
- **Training and held-out samples are drawn alike on the map.** The split is
  a number in "Prediction and loss", not a glyph to decode: the two
  regression generators are noiseless, so their splits cannot differ, and the
  classifiers' irreducible error is in the overlap of the classes rather than
  in which split a point landed in.
- There is no GitHub Pages workflow yet, though the siblings deploy from
  `main`. The build is a plain `npm run build`, so adding one is a deliberate
  step rather than a default.
- The plan proposes "a thin React interface". The spike uses plain TypeScript
  with no framework, matching `../mutual`; the numerical modules are already
  UI-independent, so adopting React later is additive. Revisit when the
  inspector grows.
