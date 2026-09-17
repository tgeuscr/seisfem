"""Localized potential packets and paired local-material boundary experiments."""

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D

LOW = dict(density=2400, vp=3200, vs=1800)
HIGH = dict(density=2800, vp=4000, vs=2200)
FREQUENCY = 8.0
TRANSVERSE_WIDTH = 400.0


def initial(x, center, direction, mode, speed):
    tangent = np.array([-direction[1], direction[0]])
    s = (x - center) @ direction
    q = (x - center) @ tangent
    w = speed / (np.pi * FREQUENCY)
    W = TRANSVERSE_WIDTH
    exponential = np.exp(-((s / w) ** 2) - (q / W) ** 2)
    fs = (1 - 2 * (s / w) ** 2) * exponential
    fq = -2 * q * s / W**2 * exponential
    fss = (-6 * s / w**2 + 4 * s**3 / w**4) * exponential
    fsq = -2 * q / W**2 * (1 - 2 * (s / w) ** 2) * exponential
    u = fs[:, None] * direction + fq[:, None] * tangent
    v = -speed * (fss[:, None] * direction + fsq[:, None] * tangent)
    if mode == "S":
        u = u[:, [1, 0]] * np.array([1, -1])
        v = v[:, [1, 0]] * np.array([1, -1])
    return u, v


def run(
    mode="P", layer=0, side="right", angle=0, h=20, representation="layered", boundary="absorbing"
):
    material = LOW if layer == 0 else HIGH
    c = material["vp" if mode == "P" else "vs"]
    theta = np.deg2rad(angle)
    direction = (
        np.array([np.cos(theta), np.sin(theta)]) if side == "right" else np.array([0.0, -1.0])
    )
    hit = (
        np.array([2400.0, -2400.0 if layer == 0 else 2400.0])
        if side == "right"
        else np.array([1200.0, -4800.0])
    )
    center = hit - 1200 * direction
    reflected = direction * np.array([-1, 1]) if side == "right" else direction * np.array([1, -1])
    receiver = hit + 400 * reflected
    lower = (0, -7200 if boundary == "reference" and side == "lower" else -4800)
    upper = (4800 if boundary == "reference" and side == "right" else 2400, 4800)
    layers = dict(
        type="layered",
        layers=[
            dict(lower=lower[1], upper=0, material=LOW),
            dict(lower=0, upper=upper[1], material=HIGH),
        ],
    )
    boundaries = dict(left="absorbing", right="absorbing", lower="absorbing", upper="absorbing")
    if boundary == "free":
        boundaries[side] = "free"
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(
                lower=lower,
                upper=upper,
                cells=(round((upper[0] - lower[0]) / h), round((upper[1] - lower[1]) / h)),
            ),
            material=layers if representation == "layered" else material,
            boundaries=boundaries,
        )
    )
    # Use identical dt across representations, safe for the fastest material.
    dt = 0.002 * (h / 20)
    reflection_time = 1600 / c
    time = np.arange(int(np.ceil((reflection_time + 0.8 / FREQUENCY) / dt)) + 1) * dt
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        u, v = initial(op.coordinates[: op.n // 2], center, direction, mode, c)
        step = op.start(dt, u0=u.ravel(), v0=v.ravel())
        points = PointMap2D(op.V, [receiver])
        traces = np.empty((len(time), 2))
        velocities = np.empty_like(traces)
        zero = np.zeros(op.n)
        for j in range(len(time)):
            nxt, velocity, _, _ = step.evaluate(zero)
            traces[j] = points.evaluate(step.current)[0]
            velocities[j] = points.evaluate(velocity)[0]
            if j < len(time) - 1:
                step.advance(nxt)
    return time, traces, velocities


def comparison(mode="P", layer=0, side="right", angle=0, h=20):
    results = {}
    for representation in ["layered", "homogeneous"]:
        traces = {
            boundary: run(mode, layer, side, angle, h, representation, boundary)
            for boundary in ["absorbing", "free", "reference"]
        }
        c = (LOW if layer == 0 else HIGH)["vp" if mode == "P" else "vs"]
        window = abs(traces["free"][0] - 1600 / c) <= 0.8 / FREQUENCY
        free = traces["free"][1] - traces["reference"][1]
        absorbed = traces["absorbing"][1] - traces["reference"][1]
        peak = np.max(np.linalg.norm(free[window], axis=1))
        residual = np.max(np.linalg.norm(absorbed[window], axis=1))
        results[representation] = dict(
            free_peak=float(peak),
            absorbing_peak=float(residual),
            reflection_proxy=float(residual / peak),
        )
        results[representation]["u"] = traces["absorbing"][1]
        results[representation]["v"] = traces["absorbing"][2]
    a, b = results["layered"], results["homogeneous"]
    differences = {
        key: float(np.max(abs(a[key] - b[key])) / max(np.max(abs(b[key])), 1e-30))
        for key in ["u", "v"]
    }
    for r in results.values():
        del r["u"], r["v"]
    return dict(
        mode=mode,
        layer=layer,
        side=side,
        angle=angle,
        h=h,
        results=results,
        representation_difference=differences,
        proxy_difference=abs(a["reflection_proxy"] - b["reflection_proxy"]),
    )
