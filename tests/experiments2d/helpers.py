"""Fixed, physical experiment definitions and independent signal diagnostics."""

import numpy as np

from seisfem import SimulationConfig2D


def small_config(**updates):
    data = dict(
        domain=dict(lower=(0, 0), upper=(1, 1), cells=(8, 8)),
        material=dict(density=2.3, vp=2.5, vs=1.2),
        time=dict(dt=0.002, duration=0.12),
        source=dict(
            position=(0.413, 0.527),
            direction=(3, -4),
            wavelet=dict(f0=9, amplitude=2.1, time_shift=0.04),
        ),
        receivers=[
            dict(name="B", position=(0.63, 0.74)),
            dict(name="A", position=(0.12, 0.39)),
            dict(name="edge", position=(0, 0.37)),
        ],
    )
    data.update(updates)
    return SimulationConfig2D.model_validate(data)


def wave_config(cells=160):
    # The central source and axial stations separate symmetry-allowed P/S signals.
    # Same physical points, source spectrum, amplitude, geometry and dt at every h.
    return SimulationConfig2D.model_validate(
        dict(
            domain=dict(
                lower=(-2400, -2400), upper=(2400, 2400), cells=(cells, cells), diagonal="right"
            ),
            material=dict(density=2400, vp=3200, vs=1800),
            time=dict(dt=0.001, duration=0.95),
            source=dict(
                position=(3.7, 2.9),
                direction=(1, 0),
                wavelet=dict(f0=8, amplitude=1e8, time_shift=0.1875),
            ),
            receivers=[
                dict(name="axial", position=(903.7, 2.9)),
                dict(name="transverse", position=(3.7, 902.9)),
                dict(name="oblique", position=(640.0961030678928, 639.2961030678928)),
            ],
        )
    )


def envelope(signal):
    """FFT analytic-signal envelope, with zero padding to reduce periodic wrap."""
    size = len(signal)
    padded = 4 * size
    spectrum = np.fft.fft(signal, n=padded)
    multiplier = np.zeros(padded)
    multiplier[0] = multiplier[padded // 2] = 1
    multiplier[1 : padded // 2] = 2
    return abs(np.fft.ifft(spectrum * multiplier)[:size])


def arrival(time, signal, predicted, half_window=0.065):
    """Windowed analytic-envelope maximum with three-sample parabolic refinement."""
    amplitude = envelope(signal)
    ids = np.flatnonzero(abs(time - predicted) <= half_window)
    index = ids[np.argmax(amplitude[ids])]
    if index == ids[0] or index == ids[-1]:
        raise AssertionError("Arrival maximum reached the diagnostic window boundary")
    left, middle, right = amplitude[index - 1 : index + 2]
    fraction = 0.5 * (left - right) / (left - 2 * middle + right)
    return float(time[index] + fraction * (time[1] - time[0]))


def arrival_report(result):
    r, t0 = 900, 0.1875
    expected = np.array([t0 + r / 3200, t0 + r / 1800])
    measured = np.array(
        [
            arrival(result.time, result.displacement[:, 0, 0], expected[0]),
            arrival(result.time, result.displacement[:, 1, 0], expected[1]),
        ]
    )
    return expected, measured


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
