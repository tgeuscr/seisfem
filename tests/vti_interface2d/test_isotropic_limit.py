import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from tests.oblique2d import diagnostics as old_diagnostics
from tests.oblique2d import experiment as old_experiment
from tests.oblique2d import packets as old_packets
from tests.oblique2d import reference as old_reference
from tests.plane_strain.helpers import dense

from . import experiment
from .conftest import record
from .reference import Material


def test_isotropic_limit_matrices():
    outputs = []
    for vti in [False, True]:
        m = old_reference.UPPER
        upper = (
            dict(
                type="vti",
                density=m.rho,
                c11=m.rho * m.vp**2,
                c33=m.rho * m.vp**2,
                c13=m.rho * (m.vp**2 - 2 * m.vs**2),
                c55=m.rho * m.vs**2,
            )
            if vti
            else dict(density=m.rho, vp=m.vp, vs=m.vs)
        )
        cfg = PlaneStrainConfig.model_validate(
            dict(
                domain=dict(lower=(-1, -1), upper=(1, 1), cells=(6, 6)),
                material=dict(
                    type="layered",
                    layers=[
                        dict(lower=-1, upper=0, material=dict(density=2200, vp=3000, vs=1700)),
                        dict(lower=0, upper=1, material=upper),
                    ],
                ),
            )
        )
        with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
            outputs.append(
                dict(M=dense(op.M), K=dense(op.K), mass=op.mass.copy(), dt=np.array(op.stable_dt))
            )
    errors = {
        key: float(np.max(abs(value - outputs[1][key])) / np.max(abs(value)))
        for key, value in outputs[0].items()
    }
    assert max(errors.values()) < 3e-15
    record("isotropic-matrices", errors)


def test_existing_isotropic_interface_experiment_through_vti(monkeypatch):
    # Reproduce the trusted experiment exactly, rather than compare different beams.
    monkeypatch.setattr(experiment, "TIME", old_packets.TIMES["P"])
    monkeypatch.setattr(experiment, "initial", lambda x, angle: old_packets.initial(x, "P"))
    m = old_reference.UPPER
    upper = Material.isotropic(m.rho, m.vp, m.vs)
    new, new_grid = experiment.run(25, 20, MPI.COMM_SELF, extent=old_packets.EXTENT, upper=upper)
    old, old_grid = old_experiment.run("P", 20, MPI.COMM_SELF)
    errors = {}
    for key in [
        "dt",
        "stable_dt",
        "steps",
        "purity",
        "mass",
        "mass_square",
        "stiffness_form",
        "action_square",
        "trace",
    ]:
        a, b = np.asarray(new[key]), np.asarray(old[key])
        errors[key] = float(np.max(abs(a - b)) / max(np.max(abs(b)), 1e-30))
    errors["final_u_v"] = float(np.max(abs(new_grid - old_grid)) / np.max(abs(old_grid)))
    assert max(errors.values()) < 3e-12
    a, b = old_diagnostics.measure(new_grid, "P", 20), old_diagnostics.measure(old_grid, "P", 20)
    for name, row in a["branches"].items():
        for key in ["amplitude", "imaginary", "flux", "angle"]:
            assert abs(row[key] - b["branches"][name][key]) < 3e-11
        assert row["signed_error"] < 0.006
        assert abs(row["imaginary"]) < 0.10
        assert row["flux_error"] < 0.014
    assert a["closure_error"] < 0.016
    record("isotropic-experiment", dict(errors=errors, diagnostics=a))
