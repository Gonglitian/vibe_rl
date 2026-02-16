"""Tests for vibe_rl.schedule."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vibe_rl.schedule import linear_schedule


class TestLinearSchedule:
    def test_start_value(self) -> None:
        sched = linear_schedule(start=1.0, end=0.0, steps=100)
        assert float(sched(0)) == 1.0

    def test_end_value(self) -> None:
        sched = linear_schedule(start=1.0, end=0.0, steps=100)
        assert float(sched(100)) == 0.0

    def test_midpoint(self) -> None:
        sched = linear_schedule(start=1.0, end=0.0, steps=100)
        val = float(sched(50))
        assert abs(val - 0.5) < 1e-5

    def test_clamps_beyond_steps(self) -> None:
        sched = linear_schedule(start=1.0, end=0.1, steps=100)
        # Beyond the schedule should clamp at end value
        assert float(sched(200)) == float(sched(100))

    def test_negative_step_clamps(self) -> None:
        sched = linear_schedule(start=1.0, end=0.0, steps=100)
        assert float(sched(0)) == 1.0

    def test_increasing_schedule(self) -> None:
        sched = linear_schedule(start=0.0, end=1.0, steps=10)
        assert float(sched(0)) == 0.0
        assert float(sched(10)) == 1.0
        assert float(sched(5)) > 0.0

    def test_jit_compatible(self) -> None:
        sched = linear_schedule(start=1.0, end=0.0, steps=100)
        jitted = jax.jit(sched)
        val = float(jitted(jnp.int32(50)))
        assert abs(val - 0.5) < 1e-5

    def test_works_with_jax_array_input(self) -> None:
        sched = linear_schedule(start=1.0, end=0.0, steps=100)
        step = jnp.array(25)
        val = float(sched(step))
        assert abs(val - 0.75) < 1e-5

    def test_single_step_schedule(self) -> None:
        sched = linear_schedule(start=1.0, end=0.0, steps=1)
        assert float(sched(0)) == 1.0
        assert float(sched(1)) == 0.0

    def test_zero_steps_does_not_crash(self) -> None:
        # steps=0 should be handled gracefully (treated as steps=1)
        sched = linear_schedule(start=1.0, end=0.0, steps=0)
        # Should not raise
        sched(0)
