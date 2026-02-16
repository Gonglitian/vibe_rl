"""Tests for the training runner.

NOTE: The runner currently expects an OOP agent with observe/act/learn/save
methods. Once the runner is adapted for the functional JAX API, these tests
should be rewritten.
"""

import pytest

pytest.skip(
    "Runner tests pending adaptation to functional JAX agent API",
    allow_module_level=True,
)
