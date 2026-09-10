"""Independent velocity Green functions and traction-continuity oracles."""

import numpy as np
import pytest

from seisfem import Simulation, SimulationConfig


def wave_config(mode="P", cells=4000, source=1000.37, receivers=(1300.23, 2700.41), pair=None):
    first = dict(density=1730, vp=1870, vs=930)
    second = dict(density=2670, vp=3210, vs=1420)
    first, second = pair or (first, second)
    return SimulationConfig.model_validate(
        dict(
            mode=mode,
            domain=dict(upper=4000),
            mesh=dict(cells=cells),
            materials=dict(
                type="layered",
                layers=[
                    dict(lower=0, upper=2000, material=first),
                    dict(lower=2000, upper=4000, material=second),
                ],
            ),
            time=dict(dt=0.0001, duration=2.8),
            source=dict(position=source, amplitude=-6.7e6, frequency=8, time_shift=0.2),
            receivers=[dict(name=f"r{i}", position=z) for i, z in enumerate(receivers)],
            boundaries=dict(lower="absorbing", upper="absorbing"),
        )
    )


def reference(time, arrival, amplitude, frequency=8):
    # Deliberately local elementary formula, no production source helper.
    a = np.pi * frequency * (time - arrival)
    return amplitude * (1 - 2 * a * a) * np.exp(-a * a)


@pytest.mark.parametrize("mode", ["P", "S"])
@pytest.mark.parametrize("reverse", [False, True])
def test_interface_both_contrast_directions(mode, reverse):
    pair = [dict(density=1730, vp=1870, vs=930), dict(density=2670, vp=3210, vs=1420)]
    if reverse:
        pair.reverse()
    cfg = wave_config(mode, cells=8000 if mode == "S" else 4000, pair=pair)
    result = Simulation(cfg).run()
    c1, c2 = [m["vp" if mode == "P" else "vs"] for m in pair]
    z1, z2 = [m["density"] * c for m, c in zip(pair, [c1, c2], strict=True)]
    # Solve continuity of velocity and signed internal stress, instead of a coefficient helper.
    R, T = np.linalg.solve([[1, -1], [z1, z2]], [-1, z1])
    assert R * R + z2 / z1 * T * T == pytest.approx(1, abs=5e-16)
    amp = -6.7e6 / (2 * z1)
    phases = [
        (0, 0.2 + (1300.23 - 1000.37) / c1, amp),
        (0, 0.2 + (4000 - 1000.37 - 1300.23) / c1, amp * R),
        (1, 0.2 + (2000 - 1000.37) / c1 + (2700.41 - 2000) / c2, amp * T),
    ]
    # Even imperfect outer absorbers must have their reflected main pulses outside
    # the comparison windows. Check path times rather than assuming absorption.
    boundary_arrivals = {
        0: [0.2 + (1000.37 + 1300.23) / c1, phases[1][1] + 4000 / c2],
        1: [phases[2][1] + 2 * 1000.37 / c1, phases[2][1] + 2 * (4000 - 2700.41) / c2],
    }
    errors = []
    for index, arrival, amplitude in phases:
        assert all(abs(arrival - other) > 2 * 0.09 for other in boundary_arrivals[index])
        mask = abs(result.time - arrival) < 0.09
        exact = reference(result.time[mask], arrival, amplitude)
        values = result.velocity[mask, index, 0]
        errors.append(np.linalg.norm(values - exact) / np.linalg.norm(exact))
        assert errors[-1] < 0.025
        measured = values[np.argmax(np.sign(amplitude) * values)]
        assert abs(measured / amplitude - 1) < 0.01
    print("independent interface", mode, reverse, "R/T", R, T, "L2", errors)


@pytest.mark.parametrize("mode", ["P", "S"])
def test_force_at_interface_two_outgoing_impedances(mode):
    cfg = wave_config(mode, cells=8000 if mode == "S" else 4000, source=2000)
    result = Simulation(cfg).run()
    c1, c2 = (1870, 3210) if mode == "P" else (930, 1420)
    # Force balances the tractions of two outgoing waves at the loaded interface.
    amp = -6.7e6 / (1730 * c1 + 2670 * c2)
    for i, travel in [(0, (2000 - 1300.23) / c1), (1, (2700.41 - 2000) / c2)]:
        arrival = 0.2 + travel
        mask = abs(result.time - arrival) < 0.09
        expected = reference(result.time[mask], arrival, amp)
        assert (
            np.linalg.norm(result.velocity[mask, i, 0] - expected) / np.linalg.norm(expected) < 0.02
        )


def test_nearly_equal_impedance_reflection_absolute_error():
    first = dict(density=1730, vp=1870, vs=930)
    second = dict(density=1730 * 1870 / 2300 * (1 + 1e-4), vp=2300, vs=1100)
    result = Simulation(wave_config(cells=8000, pair=(first, second))).run()
    z1, z2 = 1730 * 1870, second["density"] * 2300
    R, T = np.linalg.solve([[1, -1], [z1, z2]], [-1, z1])
    amp = -6.7e6 / (2 * z1)
    arrival = 0.2 + (4000 - 1000.37 - 1300.23) / 1870
    mask = abs(result.time - arrival) < 0.09
    residual = np.max(
        abs(result.velocity[mask, 0, 0] - reference(result.time[mask], arrival, amp * R))
    ) / abs(amp)
    print("near-equal impedance R, reflected absolute error / incident", R, residual)
    assert residual < 1e-4  # Absolute normalization remains meaningful as R approaches zero.
    arrival = 0.2 + (2000 - 1000.37) / 1870 + (2700.41 - 2000) / 2300
    mask = abs(result.time - arrival) < 0.09
    expected = reference(result.time[mask], arrival, amp * T)
    assert np.linalg.norm(result.velocity[mask, 1, 0] - expected) / np.linalg.norm(expected) < 0.005


@pytest.mark.parametrize("cells", [1000, 2000, 4000])
def test_source_position_amplitude_and_mesh_refinement(cells):
    material = dict(density=1730, vp=1870, vs=930)
    errors = []
    h = 4000 / cells
    for fraction in [0, 1e-8, 0.27, 1 - 1e-8]:
        source = 800 + fraction * h
        cfg = wave_config(
            cells=cells, source=source, receivers=(source + 600.173,), pair=(material, material)
        )
        result = Simulation(cfg).run()
        arrival, amp = 0.2 + 600.173 / 1870, -6.7e6 / (2 * 1730 * 1870)
        mask = abs(result.time - arrival) < 0.09
        exact = reference(result.time[mask], arrival, amp)
        error = np.linalg.norm(result.velocity[mask, 0, 0] - exact) / np.linalg.norm(exact)
        errors.append(error)
        assert error < 0.003 * (h**2)  # Observed O(h²) away from the point; fixed small dt.
        peak = np.min(result.velocity[mask, 0, 0])
        assert abs(peak / amp - 1) < 0.005
    print("source fractions", cells, errors)


@pytest.mark.parametrize(
    "mode,frequency,cfl", [("P", 6, 0.2), ("P", 12, 0.6), ("S", 6, 0.6), ("S", 12, 0.2)]
)
def test_absorber_both_endpoints_refinement(mode, frequency, cfl):
    speed = 2000 if mode == "P" else 1000
    residuals = []
    for cells in [500, 1000, 2000]:
        h = 2000 / cells
        dt = cfl * h / speed
        both = []
        for reverse in [False, True]:
            source, receiver = (800, 1200) if not reverse else (1200, 800)
            cfg = SimulationConfig.model_validate(
                dict(
                    mode=mode,
                    domain=dict(upper=2000),
                    mesh=dict(cells=cells),
                    materials=dict(
                        type="homogeneous", material=dict(density=2370, vp=2000, vs=1000)
                    ),
                    time=dict(dt=dt, duration=round((0.3 + 2400 / speed) / dt) * dt),
                    source=dict(
                        position=source,
                        amplitude=2 * 2370 * speed,
                        frequency=frequency,
                        time_shift=1.7 / frequency,
                    ),
                    receivers=[dict(name="r", position=receiver)],
                    boundaries=dict(lower="absorbing", upper="absorbing"),
                )
            )
            result = Simulation(cfg).run()
            arrival = 1.7 / frequency + 2000 / speed
            mask = abs(result.time - arrival) < 0.8 / frequency
            both.append(np.max(abs(result.velocity[mask, 0, 0])))
        np.testing.assert_allclose(both[0], both[1], rtol=1e-6, atol=3e-11)
        residuals.append(max(both))
    print("absorber sweep", mode, frequency, cfl, residuals)
    assert residuals[1] < 0.3 * residuals[0]
    assert residuals[2] < 0.3 * residuals[1]
    assert residuals[-1] < 0.003


def test_absorber_timestep_sweep_at_fixed_mesh():
    residuals = []
    for cfl in [0.2, 0.6, 0.85]:
        dt = cfl * 0.001
        cfg = SimulationConfig.model_validate(
            dict(
                domain=dict(upper=2000),
                mesh=dict(cells=1000),
                materials=dict(type="homogeneous", material=dict(density=1130, vp=2000, vs=1000)),
                time=dict(dt=dt, duration=round(1.4 / dt) * dt),
                source=dict(
                    position=800, amplitude=2 * 1130 * 2000, frequency=12, time_shift=1.7 / 12
                ),
                receivers=[dict(name="r", position=1200)],
                boundaries=dict(lower="absorbing", upper="absorbing"),
            )
        )
        result = Simulation(cfg).run()
        mask = abs(result.time - (1.7 / 12 + 1)) < 0.8 / 12
        residuals.append(np.max(abs(result.velocity[mask, 0, 0])))
    print("fixed-h absorber dt sweep", residuals)
    assert max(residuals) < 0.002
    # Smaller dt need not reduce spatial boundary mismatch: dispersion errors can cancel.
