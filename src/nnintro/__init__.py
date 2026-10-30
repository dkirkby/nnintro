"""Offline asset pipeline for the dense neural network explorer.

Trains the catalog's checkpoints in PyTorch and exports them, with golden
probe fixtures, for the browser to load. Nothing here runs in the browser:
see PLAN.md sections 7 and 8.
"""

# 0.2.0 added `dataset.label` to every checkpoint and a `datasets` array to
# the manifest, both of which the catalog controls require: a 0.1.0 snapshot
# no longer satisfies the frontend's manifest validation.
SCHEMA_VERSION = "0.2.0"

__all__ = ["SCHEMA_VERSION"]
