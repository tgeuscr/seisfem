from seisfem.config import SimulationConfig


def pulse_config(mode="P", cells=2000, layered=False, **updates):
    data = dict(
        mode=mode,
        domain=dict(lower=0, upper=4000),
        mesh=dict(cells=cells),
        materials=dict(type="homogeneous", material=dict(density=2000, vp=2000, vs=1000)),
        time=dict(dt=0.0004, duration=1.4),
        source=dict(position=1000.7, amplitude=8e6, frequency=10, time_shift=0.15),
        receivers=[
            dict(name="incident", position=1400.3),
            dict(name="transmitted", position=2600.3),
        ],
        boundaries=dict(lower="absorbing", upper="absorbing"),
    )
    if layered:
        data["materials"] = dict(
            type="layered",
            layers=[
                dict(lower=0, upper=2000, material=dict(density=2000, vp=2000, vs=1000)),
                dict(lower=2000, upper=4000, material=dict(density=2400, vp=3000, vs=1500)),
            ],
        )
    data.update(updates)
    return SimulationConfig.model_validate(data)
