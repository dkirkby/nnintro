"""Offline asset pipeline for the dense neural network explorer.

Trains the catalog's checkpoints in PyTorch and exports them, with golden
probe fixtures, for the browser to load. Nothing here runs in the browser:
see PLAN.md sections 7 and 8.
"""

# 0.2.0 added `dataset.label` to every checkpoint and a `datasets` array to
# the manifest, both of which the catalog controls require: a 0.1.0 snapshot
# no longer satisfies the frontend's manifest validation.
# 0.3.0 added `dataset.targetRange`, the generator's own range of targets
# over the domain, which is what the browser's output colour scale spans. A
# 0.2.0 snapshot still loads -- the browser falls back to the range of the
# drawn samples -- but its colour scale means something slightly different,
# so the two are distinguishable rather than silently interchangeable.
SCHEMA_VERSION = "0.3.0"

__all__ = ["SCHEMA_VERSION"]
