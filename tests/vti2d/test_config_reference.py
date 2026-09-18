import numpy as np
import pytest

from seisfem.config import Isotropic, Layered
from seisfem.vti import VTI as Material

from .helpers import ISO, RHO, VTI, C, config, record
from .reference import mode, symbol


@pytest.mark.parametrize(
    "updates",
    [
        dict(density=0),
        dict(c11=0),
        dict(c33=-1),
        dict(c55=0),
        dict(c13=np.sqrt(C[0] * C[1])),
        dict(c13=-np.sqrt(C[0] * C[1])),
        dict(c13=100e9),
        dict(c11=float("inf")),
        dict(c13=float("nan")),
        dict(c55=1e-20),
        dict(density=1e-300),
        dict(c11=1e308),
        dict(c11=1e-300, c33=1e-300, c13=0, c55=1e300),
        dict(c66=7e9),
        dict(gamma=0),
        dict(epsilon=0.1),
        dict(type="tti"),
    ],
)
def test_invalid_material(updates):
    with pytest.raises(ValueError):
        config(material=VTI | updates)


def test_parsing_compatibility_and_boundary_rejection():
    assert isinstance(config(material=ISO).material, Isotropic)
    iso_layers = dict(type="layered", layers=[dict(lower=-1, upper=1, material=ISO)])
    old = config(material=iso_layers)
    assert type(old.material) is Layered
    assert not old.has_vti
    for layered in [False, True]:
        cfg = config(layered)
        assert cfg.has_vti
        assert type(cfg).model_validate_json(cfg.model_dump_json(by_alias=True)) == cfg
        assert type(cfg).model_validate(cfg.model_dump(by_alias=True)) == cfg
        for side in ["left", "right", "lower", "upper"]:
            with pytest.raises(ValueError, match="VTI absorbing boundaries are not validated"):
                config(layered, boundaries={side: "absorbing"})
            assert config(layered, constraints=[dict(side=side, components=["x"])])
    with pytest.raises(ValueError):
        Material.model_validate({k: v for k, v in VTI.items() if k != "type"})
    # The new constitutive model cannot leak into the scalar 1D configuration.
    with pytest.raises(ValueError):
        Layered.model_validate(config(True).material.model_dump(by_alias=True))


@pytest.mark.parametrize("theta", [0, 15, 35, 60, 90, -15, -35, -60, -90])
def test_christoffel_eigenproblem_and_group_derivative(theta):
    n = np.array([np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))])
    matrix = symbol(n, C)
    a, b, d = matrix[0, 0], matrix[0, 1], matrix[1, 1]
    exact = np.sqrt(
        np.array([a + d - np.hypot(a - d, 2 * b), a + d + np.hypot(a - d, 2 * b)]) / (2 * RHO)
    )
    errors = {}
    for branch, j in [("S", 0), ("P", 1)]:
        speed, pol, group = mode(n, RHO, C, branch)
        residual = np.linalg.norm(matrix @ pol - RHO * speed**2 * pol) / np.linalg.norm(matrix)
        assert residual < 1e-15
        assert abs(speed / exact[j] - 1) < 1e-15
        derivative = np.array(
            [
                (mode(n + 1e-5 * e, RHO, C, branch)[0] - mode(n - 1e-5 * e, RHO, C, branch)[0])
                / 2e-5
                for e in np.eye(2)
            ]
        )
        np.testing.assert_allclose(group, derivative, rtol=2e-9, atol=1e-7)
        errors[branch] = dict(
            speed=float(speed),
            polarization=pol.tolist(),
            eigen_residual=float(residual),
            group=group.tolist(),
        )
    record(f"reference-{theta}", errors)


def test_axis_values_and_isotropic_limit():
    for n, expected in [
        ([0, 1], [np.sqrt(C[1] / RHO), np.sqrt(C[3] / RHO)]),
        ([1, 0], [np.sqrt(C[0] / RHO), np.sqrt(C[3] / RHO)]),
    ]:
        for branch, speed in zip(["P", "S"], expected, strict=True):
            v, d, _ = mode(n, RHO, C, branch)
            assert abs(v / speed - 1) < 1e-15
            target = np.array(n) if branch == "P" else np.array([n[1], -n[0]])
            assert abs(abs(d @ target) - 1) < 1e-15
    lam, mu = (
        ISO["density"] * (ISO["vp"] ** 2 - 2 * ISO["vs"] ** 2),
        ISO["density"] * ISO["vs"] ** 2,
    )
    for theta in np.linspace(-np.pi, np.pi, 25):
        n = np.array([np.sin(theta), np.cos(theta)])
        for branch in ["P", "S"]:
            v, d, g = mode(n, ISO["density"], (lam + 2 * mu, lam + 2 * mu, lam, mu), branch)
            speed = ISO["vp" if branch == "P" else "vs"]
            np.testing.assert_allclose(v, speed, rtol=1e-15)
            np.testing.assert_allclose(g, speed * n, rtol=1e-14, atol=1e-11)
            assert abs(abs(d @ n) - (1 if branch == "P" else 0)) < 1e-14


def test_negative_c13_and_misaligned_layer_rejection():
    assert config(material=VTI | dict(c13=-9e9)).has_vti
    assert config(material=VTI | dict(c13=0)).has_vti
    data = config(True).model_dump(mode="json", by_alias=True)
    data["material"]["layers"][0]["upper"] = 0.1
    data["material"]["layers"][1]["lower"] = 0.1
    with pytest.raises(ValueError, match="horizontal mesh row"):
        type(config()).model_validate(data)
