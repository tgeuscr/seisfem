"""Central differences on owned arrays; no finite-element or frontend dependencies."""

from collections.abc import Callable

import numpy as np


class CentralDifference:
    """Second-order update with diagonal mass/damping and fixed-zero projection.

    `stiffness` may be collective. All arrays use the same owned-DOF ordering.
    Initial displacement and velocity are SI quantities; force is the assembled
    weak load. Runtime initial data are exposed here for verification, while
    declarative simulations currently use zero initial data.
    """

    def __init__(
        self,
        mass: np.ndarray,
        damping: np.ndarray,
        fixed: np.ndarray,
        dt: float,
        stiffness: Callable,
        u0: np.ndarray,
        v0: np.ndarray,
        force0: np.ndarray,
    ):
        self.mass, self.damping, self.fixed = mass, damping, fixed
        self.dt, self.stiffness = dt, stiffness
        self.current = u0.copy()
        self.current[fixed] = 0
        velocity = v0.copy()
        velocity[fixed] = 0
        a0 = (force0 - stiffness(self.current) - damping * velocity) / mass
        a0[fixed] = 0
        self.previous = self.current - dt * velocity + 0.5 * dt**2 * a0

    def evaluate(
        self, force: np.ndarray, energy: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
        """Return next state, centered velocity/acceleration and local half-step energy."""
        dt, m, c = self.dt, self.mass, self.damping
        ku = self.stiffness(self.current)
        nxt = (2 * m * self.current - (m - dt * c / 2) * self.previous + dt**2 * (force - ku)) / (
            m + dt * c / 2
        )
        nxt[self.fixed] = 0
        velocity = (nxt - self.previous) / (2 * dt)
        acceleration = (nxt - 2 * self.current + self.previous) / dt**2
        half_energy = None
        if energy:
            half_v = (nxt - self.current) / dt
            half_energy = float(0.5 * np.dot(m * half_v, half_v) + 0.5 * np.dot(nxt, ku))
        return nxt, velocity, acceleration, half_energy

    def advance(self, nxt: np.ndarray) -> None:
        """Accept an evaluated state; callers own sampling and time bookkeeping."""
        self.previous, self.current = self.current, nxt
