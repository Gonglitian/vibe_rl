"""Root test configuration.

Sets XLA_FLAGS *before* JAX is imported anywhere so that multi-GPU
tests can simulate 4 devices on a single-CPU CI runner.  This must
live in conftest.py (loaded by pytest before any test module) because
setting the flag after JAX's backend initialises has no effect.
"""

import os

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)
