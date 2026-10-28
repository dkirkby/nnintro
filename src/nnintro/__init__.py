"""Offline asset pipeline for the dense neural network explorer.

Trains the catalog's checkpoints in PyTorch and exports them, with golden
probe fixtures, for the browser to load. Nothing here runs in the browser:
see PLAN.md sections 7 and 8.
"""

SCHEMA_VERSION = "0.1.0"

__all__ = ["SCHEMA_VERSION"]
