import numpy as np
import pytest

from seisfem import Simulation
from seisfem.sources import ricker
from tests.helpers import pulse_config

pytestmark = pytest.mark.analytical


def pulse_metrics(result, index, arrival, amplitude, frequency=10):
    time = result.time
    values = result.velocity[:, index, 0]
    window = abs(time - arrival) < 0.07
    reference = amplitude * ricker(time[window], frequency, arrival)
    relative_error = np.linalg.norm(values[window] - reference) / np.linalg.norm(reference)
    peak_index = np.argmax(np.sign(amplitude) * values[window])
    return (
        time[window][peak_index] - arrival,
        values[window][peak_index] / amplitude - 1,
        relative_error,
    )


@pytest.mark.parametrize("mode,c", [("P", 2000), ("S", 1000)])
def test_homogeneous_speed_and_amplitude(mode, c):
    cfg = pulse_config(
        mode,
        cells=2000 if mode == "P" else 8000,
        time=dict(dt=0.0002, duration=1.4),
        receivers=[dict(name="r", position=1600.3)],
    )
    result = Simulation(cfg).run()
    arrival = 0.15 + (1600.3 - 1000.7) / c
    metrics = pulse_metrics(result, 0, arrival, 8e6 / (2 * 2000 * c))
    print(mode, "arrival error, relative amplitude error, waveform L2:", metrics)
    # 2 ms bounds subcell sampling plus resolved P1 phase error (>=16 cells/lambda).
    assert abs(metrics[0]) < 0.002
    assert abs(metrics[1]) < 0.015
    assert metrics[2] < 0.02
    # Independently check the integrated and differentiated Green functions.
    mask = abs(result.time - arrival) < 0.07
    tau = result.time[mask] - arrival
    a = np.pi * 10 * tau
    amplitude = 8e6 / (2 * 2000 * c)
    displacement = amplitude * (tau * np.exp(-(a**2)) + 0.15 * np.exp(-((np.pi * 10 * 0.15) ** 2)))
    acceleration = amplitude * np.pi * 10 * (4 * a**3 - 6 * a) * np.exp(-(a**2))
    for quantity, reference in (("displacement", displacement), ("acceleration", acceleration)):
        error = np.linalg.norm(getattr(result, quantity)[mask, 0, 0] - reference)
        relative = error / np.linalg.norm(reference)
        print(mode, quantity, "relative waveform error", relative)
        assert relative < 0.025


@pytest.mark.parametrize("mode,c1,c2", [("P", 2000, 3000), ("S", 1000, 1500)])
def test_layered_interface(mode, c1, c2):
    cfg = pulse_config(
        mode, cells=2000 if mode == "P" else 8000, layered=True, time=dict(dt=0.0002, duration=2.2)
    )
    result = Simulation(cfg).run()
    z1, z2 = 2000 * c1, 2400 * c2
    reflection, transmission = (z1 - z2) / (z1 + z2), 2 * z1 / (z1 + z2)
    assert reflection**2 + z2 / z1 * transmission**2 == pytest.approx(1)
    incident = 8e6 / (2 * z1)
    arrivals = [
        0.15 + (1400.3 - 1000.7) / c1,
        0.15 + ((2000 - 1000.7) + (2000 - 1400.3)) / c1,
        0.15 + (2000 - 1000.7) / c1 + (2600.3 - 2000) / c2,
    ]
    for index, arrival, amplitude in zip(
        [0, 0, 1], arrivals, [incident, incident * reflection, incident * transmission], strict=True
    ):
        metrics = pulse_metrics(result, index, arrival, amplitude)
        print(mode, "layer arrival", arrival, "metrics", metrics)
        assert abs(metrics[0]) < 0.0025
        assert abs(metrics[1]) < 0.025
        assert metrics[2] < 0.035


def test_source_and_waveform_refinement():
    errors = []
    for cells in (500, 1000, 2000):
        cfg = pulse_config(
            cells=cells,
            receivers=[dict(name="r", position=1600.3)],
            time=dict(dt=0.2 * (4000 / cells) / 2000, duration=0.8),
        )
        result = Simulation(cfg).run()
        errors.append(pulse_metrics(result, 0, 0.15 + (1600.3 - 1000.7) / 2000, 1)[2])
    print("pulse refinement L2", errors)
    # Point forcing is singular: require improvement, not smooth-field asymptotics.
    assert errors[1] < 0.5 * errors[0]
    assert errors[2] < 0.5 * errors[1]


@pytest.mark.parametrize("boundary,coefficient", [("free", 1), ("fixed", -1), ("absorbing", 0)])
def test_endpoint_reflection(boundary, coefficient):
    cfg = pulse_config(
        domain=dict(lower=0, upper=2000),
        mesh=dict(cells=1000),
        source=dict(position=800, amplitude=8e6, frequency=10, time_shift=0.15),
        receivers=[dict(name="r", position=1200)],
        boundaries=dict(lower="absorbing", upper=boundary),
        time=dict(dt=0.0004, duration=1.3),
    )
    result = Simulation(cfg).run()
    arrival = 0.15 + (1200 + 800) / 2000
    if coefficient:
        metrics = pulse_metrics(result, 0, arrival, coefficient)
        print(boundary, metrics)
        assert abs(metrics[0]) < 0.002
        assert abs(metrics[1]) < 0.01
        assert metrics[2] < 0.02
    else:
        reflected = max(abs(result.velocity[abs(result.time - arrival) < 0.07, 0, 0]))
        print("absorbing residual reflection", reflected)
        assert reflected < 0.003  # <0.3% of unit incident velocity at this resolution.
