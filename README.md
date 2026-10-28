# nnintro — dense neural network explorer

An interactive explorer for students with a science background: load a trained
or random model, follow one input through the network, edit a single weight or
bias, and compare what the local derivative predicted against what actually
changed.

[`PLAN.md`](PLAN.md) is the design spec. **Stage 0 (the numerical and visual
spike) is implemented**: one dataset, one network, the PyTorch export, the
nnet-svg adapter, editable weights and biases, and the downstream pulse.
Stages 1–2 complete the initial release.

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
uv run nnintro verify                # re-check the published snapshot
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
dependency: the one it uses is `@dkirkby/zipline`.

No linter or formatter is configured, matching the sibling course projects.
`data/` is gitignored working output; `frontend/public/data/` is the tracked
snapshot the site is built from, and `nnintro publish` is the only thing that
should write it.

## The export contract

`uv run nnintro train` writes one JSON file per checkpoint plus a
`manifest.json`. The browser enumerates the manifest rather than assuming the
catalog is covered: a control must never be able to reach a configuration that
has no checkpoint.

A checkpoint holds:

| Field | Contents |
| --- | --- |
| `schemaVersion`, `checkpointId`, `contentHash` | provenance; the hash is SHA-256 over the canonical JSON with the hash field removed |
| `dataset` | id, task, domain, input/target names, generator version, and the fixed seeded samples with their split membership |
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

- Stage 0 trains **one** catalog entry (`tilted-plane`, `2-4-1`, ReLU) rather
  than all 76. `CATALOG` in `cli.py` is the list to grow in Stage 1.
- Training is followed by a **curation** step the plan calls for in section 8
  but does not specify: `train.curate` keeps the best of 24 attempts that
  passes `SELECTION_CRITERIA`. Without it, training reliably produces a model
  that fits the plane perfectly through a single hidden unit and teaches
  nothing. The chosen model's per-unit diagnostics ship under `selection`.
- There is no GitHub Pages workflow yet, though the siblings deploy from
  `main`. The build is a plain `npm run build`, so adding one is a deliberate
  step rather than a default.
- The plan proposes "a thin React interface". The spike uses plain TypeScript
  with no framework, matching `../mutual`; the numerical modules are already
  UI-independent, so adopting React later is additive. Revisit when the
  inspector grows.
