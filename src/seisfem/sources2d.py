"""Precomputed spatial vector force, separate from its callable time function."""

from .points2d import PointMap2D


class PointForce2D:
    """F(t)*b_s, where b_sᵀv = d_hat·v(xs) and F is a line force [N/m].

    Multiple instances can be summed without changing spatial discretization or
    the time integrator. The first declarative workflow exposes one source.
    """

    def __init__(self, V, config):
        self.points = PointMap2D(V, [config.position])
        self.spatial_load = self.points.unit_load(config.direction)
        self.wavelet = config.wavelet

    def __call__(self, time):
        return self.wavelet(time) * self.spatial_load
