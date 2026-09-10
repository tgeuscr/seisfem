import numpy as np
import pytest
from pydantic import ValidationError

from seisfem.config import Isotropic, SimulationConfig
from seisfem.sources import ricker


def basic():
    return dict(
        domain=dict(upper=1000),
        mesh=dict(cells=100),
        materials=dict(type="homogeneous", material=dict(density=2000, vp=2000, vs=1000)),
        time=dict(dt=0.001, duration=0.1),
    )


def test_material():
    m = Isotropic(density=2000, vp=2000, vs=1000)
    assert m.lame == (4e9, 2e9)
    other = Isotropic.model_validate(dict(density=2000, **{"lambda": 4e9}, mu=2e9))
    assert other.speed("P") == 2000
    assert other.speed("S") == 1000
    # Negative lambda is legal when bulk modulus stays positive.
    assert Isotropic.model_validate(dict(density=1, **{"lambda": -0.5}, mu=1)).speed("P") > 0
    for values in [
        dict(density=1, vp=1, vs=1),
        dict(density=0, vp=2, vs=1),
        dict(density=1, vp=2, vs=1, mu=1),
        dict(density=1, vp=2),
        dict(density=1, **{"lambda": -1}, mu=1),
        dict(density=float("nan"), vp=2, vs=1),
    ]:
        with pytest.raises(ValidationError):
            Isotropic.model_validate(values)


def test_ricker():
    f, t0 = 5, 0.4
    assert ricker(t0, f, t0) == 1
    delta = np.linspace(0, 1, 50)
    np.testing.assert_allclose(ricker(t0 + delta, f, t0), ricker(t0 - delta, f, t0), atol=1e-15)
    assert abs(ricker(t0 + 1 / (np.sqrt(2) * np.pi * f), f, t0)) < 1e-14
    # Numerically confirm frequency convention and zero mean independently.
    t = np.arange(-2, 2, 0.0005)
    wave = ricker(t, f, 0)
    frequency = np.fft.rfftfreq(len(t), 0.0005)
    assert frequency[np.argmax(abs(np.fft.rfft(wave)))] == pytest.approx(f)
    assert abs(np.trapezoid(wave, t)) < 1e-14
    with pytest.raises(ValueError):
        ricker(0, 0, 0)


def test_roundtrip(tmp_path):
    config = SimulationConfig.model_validate(basic())
    config.to_yaml(tmp_path / "config.yaml")
    assert SimulationConfig.from_yaml(tmp_path / "config.yaml") == config
    with pytest.raises(ValidationError):
        config.mode = "S"


@pytest.mark.parametrize(
    "section,change",
    [
        ("fem", dict(degree=2)),
        ("time", dict(dt=0.1, duration=1)),
        ("time", dict(dt=0.001, duration=0.0015)),
        ("domain", dict(upper=-1)),
        ("mesh", dict(cells=True)),
        ("source", dict(position=1001, frequency=5, time_shift=0.3)),
    ],
)
def test_invalid(section, change):
    data = basic()
    data[section] = change
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(data)


@pytest.mark.parametrize(
    "bad_materials",
    [
        {
            "type": "layered",
            "layers": [
                {"lower": 0, "upper": 505, "material": {"density": 1, "vp": 2, "vs": 1}},
                {"lower": 505, "upper": 1000, "material": {"density": 1, "vp": 2, "vs": 1}},
            ],
        },
        {
            "type": "layered",
            "layers": [
                {"lower": 0, "upper": 500, "material": {"density": 1, "vp": 2, "vs": 1}},
                {"lower": 600, "upper": 1000, "material": {"density": 1, "vp": 2, "vs": 1}},
            ],
        },
        {
            "type": "layered",
            "layers": [{"lower": 0, "upper": 800, "material": {"density": 1, "vp": 2, "vs": 1}}],
        },
    ],
)
def test_reject_unresolved_layers(bad_materials):
    data = basic()
    data["materials"] = bad_materials
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(data)


def test_yaml_rejects_duplicate_and_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("mode: P\nmode: S\n")
    with pytest.raises(ValueError, match="Duplicate"):
        SimulationConfig.from_yaml(path)
    data = basic()
    data["typo"] = 1
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(data)


def test_resolution_warning_does_not_replace_stability():
    data = basic()
    data["source"] = dict(position=500, frequency=30, time_shift=0.1)
    cfg = SimulationConfig.model_validate(data)
    assert cfg.diagnostics()["warnings"]
    assert cfg.time.dt < cfg.stable_dt
