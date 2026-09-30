"""Spatial complex projections and angle measurements independent of ray predictions."""

import numpy as np

from .reference import NAMES, report


def frequency_isolate(displacement, velocity, target, other):
    """Remove the other positive-frequency mode without projecting polarization.

    At fixed k, U=Ub+Uo and i*V=omega_b*Ub+omega_o*Uo. Thus the
    target vector is (i*V-omega_o*U)/(omega_b-omega_o). This retains any
    perpendicular component at the target frequency rather than zeroing it.
    """
    return (1j * velocity - other * displacement) / (target - other)


def angles(grid, packet, side, kind):
    h = packet.sampling
    size = len(grid)
    axis = np.linspace(-packet.analysis, packet.analysis, size)
    window = np.sin(np.pi / 2 * np.clip((side * axis - 120) / 360, 0, 1)) ** 2
    pad = 2 * size
    U = np.fft.fft2(grid * window[:, None, None], s=(pad, pad), axes=(0, 1))
    k = 2 * np.pi * np.fft.fftfreq(pad, h)
    kx, kz = np.meshgrid(k, k)
    radius = np.hypot(kx, kz)
    n = np.stack((kx, kz), axis=-1) / np.maximum(radius[:, :, None], 1e-30)
    d = n if kind == "P" else n[:, :, [1, 0]] * [1, -1]
    speed = (packet.lower if side < 0 else packet.upper).speed(kind)
    # Positive temporal frequency distinguishes an outgoing wave from the
    # negative-frequency conjugate of a remaining incoming packet. It uses
    # modal dispersion at every k, not a predicted Snell direction.
    positive = (U[:, :, :2] + 1j * U[:, :, 2:] / np.maximum(speed * radius[:, :, None], 1e-30)) / 2
    power = abs(np.sum(positive * d, axis=-1)) ** 2
    f = speed * radius / (2 * np.pi)
    power = np.where(
        (side * kz > 0) & (f > packet.frequency / 2) & (f < 1.5 * packet.frequency), power, 0
    )
    eligible = power > 0.02 * power.max()
    selected = np.zeros(power.shape, bool)
    pending = [np.unravel_index(np.argmax(power), power.shape)]
    while pending:
        iz, ix = pending.pop()
        if selected[iz, ix] or not eligible[iz, ix]:
            continue
        selected[iz, ix] = True
        for dz, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            pending.append(((iz + dz) % pad, (ix + dx) % pad))
    power = np.where(selected, power, 0)
    power /= power.sum()
    theta = np.degrees(np.arctan2(kx, abs(kz)))
    mean = float(np.sum(power * theta))
    return mean, float(np.sqrt(np.sum(power * (theta - mean) ** 2)))


def measure(grid, packet, templates=None):
    templates = packet.templates() if templates is None else templates
    ref = report(packet.mode, packet.angle, packet.lower, packet.upper)
    axis = np.linspace(-packet.analysis, packet.analysis, len(grid))
    omega = 2 * np.pi * packet.frequency

    def analytic(g):
        return (g[:, :, :2] + 1j * g[:, :, 2:] / omega) / 2

    measured = analytic(grid)
    xx, zz = np.meshgrid(axis, axis)
    u0, v0 = packet.initial(np.column_stack((xx.ravel(), zz.ravel())))
    incident = ((u0 + 1j * v0 / omega) / 2).reshape(measured.shape)

    def transform(field, wavevector):
        return np.einsum(
            "z,zxc,x->c",
            np.exp(-1j * wavevector[1] * axis),
            field,
            np.exp(-1j * wavevector[0] * axis),
        )

    di = packet.n if packet.mode == "P" else packet.transverse
    normalization = transform(incident, packet.k0 * packet.n) @ di
    branches = {}
    overlaps = {}
    reconstruction = np.zeros_like(measured)
    # Fit both vector modes jointly in each half-space. No fitted position,
    # angle, frequency, width or delay; only two complex scalar coefficients.
    for side, names in [(-1, NAMES[:2]), (1, NAMES[2:])]:
        mask = side * axis > 120
        columns = np.column_stack([analytic(templates[name])[mask].ravel() for name in names])
        overlaps[str(side)] = float(
            abs(np.vdot(columns[:, 0], columns[:, 1]))
            / (np.linalg.norm(columns[:, 0]) * np.linalg.norm(columns[:, 1]))
        )
        coefficients = np.linalg.lstsq(columns, measured[mask].ravel(), rcond=1e-12)[0]
        for name, coefficient in zip(names, coefficients, strict=True):
            b = ref["branches"][name]
            estimate = coefficient * b["amplitude"]
            reconstruction[mask] += coefficient * analytic(templates[name])[mask]
            angle, spread = angles(grid, packet, side, name[1])
            continuum_angle, _ = angles(templates[name], packet, side, name[1])
            d = np.array(b["polarization"])
            n = np.array(b["direction"])
            speed = (packet.lower if side < 0 else packet.upper).speed(name[1])
            ex = np.exp(-1j * omega / speed * n[0] * axis)
            ez = np.exp(-1j * omega / speed * n[1] * axis) * mask
            central = np.einsum("z,zxc,x->c", ez, measured, ex)
            jacobian = packet.speed * packet.n[1] / (speed * abs(n[1]))
            transfer = (central @ d) / normalization * jacobian * np.exp(1j * omega * packet.time)
            transverse = d[[1, 0]] * [1, -1]
            raw_overlap = float(abs(central @ transverse) / max(abs(central @ d), 1e-30))
            u_hat = np.einsum("z,zxc,x->c", ez, grid[:, :, :2], ex)
            v_hat = np.einsum("z,zxc,x->c", ez, grid[:, :, 2:], ex)
            material = packet.lower if side < 0 else packet.upper
            other_speed = material.speed("S" if name[1] == "P" else "P")
            target = frequency_isolate(u_hat, v_hat, omega, omega * other_speed / speed)
            leakage = float(abs(target @ transverse) / max(abs(target @ d), 1e-30))
            field = analytic(templates[name])[mask]
            # This template is the independently synthesized finite beam, not
            # a Gaussian translated along a central ray.
            isolated = measured[mask].copy()
            other = names[1] if name == names[0] else names[0]
            isolated -= coefficients[names.index(other)] * analytic(templates[other])[mask]
            waveform = float(np.linalg.norm(isolated - field) / max(np.linalg.norm(field), 1e-30))
            flux = b["flux_factor"] * abs(estimate) ** 2
            branches[name] = dict(
                amplitude=float(estimate.real),
                imaginary=float(estimate.imag),
                magnitude=float(abs(estimate)),
                phase_residual=float(np.angle(coefficient)),
                signed_error=float(abs(estimate.real - b["amplitude"])),
                complex_error=float(abs(estimate - b["amplitude"])),
                relative_error=float(abs(coefficient - 1)),
                angle=angle,
                spread=spread,
                angle_error=abs(angle - b["angle"]),
                continuum_angle=continuum_angle,
                angle_discretization_error=abs(angle - continuum_angle),
                polarization_leakage=leakage,
                raw_polarization_overlap=raw_overlap,
                waveform_error=waveform,
                spectral_waveform_error=modal_waveform_error(
                    grid, templates[name], packet, side, name[1]
                ),
                flux=float(flux),
                flux_error=float(abs(flux - b["flux"])),
                central_real=float(transfer.real),
                central_imaginary=float(transfer.imag),
                central_error=float(abs(transfer - b["amplitude"])),
                central_flux=float(b["flux_factor"] * abs(transfer) ** 2),
            )
    total = sum(b["flux"] for b in branches.values())
    mask = abs(axis) > 120
    return dict(
        branches=branches,
        flux_sum=float(total),
        closure_error=float(abs(total - 1)),
        template_overlap=overlaps,
        central_flux_sum=float(sum(b["central_flux"] for b in branches.values())),
        projection_residual=float(
            np.linalg.norm((measured - reconstruction)[mask]) / np.linalg.norm(measured[mask])
        ),
    )


def arrivals(row, packet):
    """Receiver-envelope timing against exact finite-packet traces, no fitted lag."""
    from tests.sem2d.packets import peak_time

    times = np.asarray(row["times"])
    window = abs(times - row["receiver_arrival"]) < 0.2
    times = times[window]
    numeric = np.asarray(row["trace"])[window]
    exact = packet.traces(row["receiver_positions"], times)
    omega = 2 * np.pi * packet.frequency
    for j, name in enumerate(NAMES):
        d = np.array(row["geometry"]["reference"]["branches"][name]["polarization"])
        a = (numeric[:, j, :2] + 1j * numeric[:, j, 2:] / omega) @ d
        b = (exact[:, j, :2] + 1j * exact[:, j, 2:] / omega) @ d
        resolved = 0 < np.argmax(abs(a)) < len(times) - 1
        observed = peak_time(times, abs(a)) if resolved else None
        expected = peak_time(times, abs(b))
        row["branches"][name].update(
            arrival=observed,
            continuum_arrival=expected,
            arrival_error=abs(observed - expected) if resolved else None,
            arrival_resolved=bool(resolved),
            ray_arrival=row["receiver_arrival"],
            receiver_waveform_error=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        )
    return row


def modal_waveform_error(grid, template, packet, side, kind):
    """Vector-mode Fourier error in the same broad outgoing band as angles.

    Unlike subtracting a fitted template of the other mode, this excludes that
    mode's numerical dispersion residual. Global residuals remain reported too.
    """
    size = len(grid)
    axis = np.linspace(-packet.analysis, packet.analysis, size)
    window = np.sin(np.pi / 2 * np.clip((side * axis - 120) / 360, 0, 1)) ** 2
    k = 2 * np.pi * np.fft.fftfreq(size, packet.sampling)
    kx, kz = np.meshgrid(k, k)
    radius = np.hypot(kx, kz)
    direction = np.stack((kx, kz), axis=-1) / np.maximum(radius[:, :, None], 1e-30)
    d = direction if kind == "P" else direction[:, :, [1, 0]] * [1, -1]
    speed = (packet.lower if side < 0 else packet.upper).speed(kind)
    omega = speed * radius
    mask = (
        (side * kz > 0)
        & (omega > np.pi * packet.frequency)
        & (omega < 3 * np.pi * packet.frequency)
    )

    def project(g):
        f = np.fft.fft2(g * window[:, None, None], axes=(0, 1))
        positive = (f[:, :, :2] + 1j * f[:, :, 2:] / np.maximum(omega[:, :, None], 1e-30)) / 2
        return np.sum(positive * d, axis=-1)[mask]

    a, b = project(grid), project(template)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
