"""Render auditable numerical tables from measured data, without solver imports."""

from .reference import NAMES


def quantitative(records, audit):
    lines = [
        "## Quantitative results",
        "",
        "Tables are generated from the companion JSON; S means SV.",
        "",
    ]

    def paragraph(text):
        lines.extend([text, ""])

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        lines.append("")

    def maximum(r, key):
        return max(b[key] for b in r["branches"].values())

    paragraph("### Materials, resolution and reference audit")
    table(
        ["Material", "rho (kg/m³)", "Vp / Vs (m/s)", "lambda / mu (GPa)"],
        [
            ["A", 2200, "3000 / 1700", "7.084 / 6.358"],
            ["B", 2500, "4000 / 2300", "13.550 / 13.225"],
        ],
    )
    paragraph(
        "P incidence is 25°; SV incidence is 15°. All four branches are propagating. "
        "For A→B the nearest central critical limit is SV incidence at asin(1700/4000) "
        "=25.15°. No critical-angle result is accepted. The same angles are used for B→A."
    )
    table(
        ["Independent reference audit", "Maximum absolute residual"],
        [[k, f"{v:.3e}"] for k, v in audit.items()],
    )
    paragraph(
        "The audit spans P/S, both material directions, and 0°, 15°, 25°, −15°. "
        "The old independent reference uses the same deterministic polarization convention; "
        "no fitted sign transformations are applied. At normal incidence the converted "
        "coefficients are exactly zero. Normal P has R=0.2048192771, T=0.7951807229; "
        "normal S has R=0.2118018967, T=0.7881981033."
    )
    keys = ["P-300", "P-200", "S-300", "S-200", "S-150", "reverse-P", "reverse-S"]
    table(
        [
            "Case",
            "h / p (m)",
            "vector DOFs",
            "dt (s)",
            "dt/safe_dt",
            "carrier SV points/λ",
            "time (s)",
        ],
        [
            [
                k,
                f"{records[k]['h']:g} / {records[k]['degree']}",
                records[k]["dofs"],
                f"{records[k]['dt']:.6g}",
                f"{records[k]['dt_safe_ratio']:.4f}",
                f"{records[k]['points_per_wavelength_S']:.2f}",
                f"{records[k]['time']:.6g}",
            ]
            for k in keys
        ],
    )
    paragraph(
        "Carrier minimum wavelengths are 600 m (P) and 340 m (SV). Average nodal spacing "
        "is h/p, not the smallest nonuniform GLL spacing. The broad analysis band extends "
        "to 7.5 Hz, so its shortest SV wavelength is 226.7 m. The 300 m SV grid is "
        "deliberately retained as under-resolution evidence. Time steps are about 0.5 ms; "
        "central-difference carrier phase accumulation is about 0.00073 rad (P) and "
        "0.0010 rad (SV), which matters at the finest spatial resolution. No fitted "
        "convergence order or exclusively spatial error claim is made at that floor."
    )
    paragraph("### Finite-beam and estimator audit")
    table(
        [
            "Incidence",
            "sigma_s / sigma_r (m)",
            "angular RMS (°)",
            "excluded spectral energy",
            "max complex recovery error",
            "max independent Fourier error",
            "inferred closure error",
        ],
        [
            [
                m,
                f"{records[m + '-200']['sigma_s']:g} / 1500",
                f"{records['continuum-' + m]['bandwidth']['spread']:.6f}",
                f"{records['continuum-' + m]['bandwidth']['excluded_energy']:.3e}",
                f"{maximum(records['continuum-' + m], 'complex_error'):.3e}",
                f"{maximum(records['continuum-' + m], 'central_error'):.3e}",
                f"{records['continuum-' + m]['closure_error']:.3e}",
            ]
            for m in ["P", "S"]
        ],
    )
    table(
        ["Incidence", "integrated-flux vector error: sigma_r=750 m", "sigma_r=1500 m"],
        [
            [m, *[f"{x:.3e}" for x in records["continuum-" + m]["width_study"]["flux_errors"]]]
            for m in ["P", "S"]
        ],
    )
    paragraph(
        "These are continuum spectrum-integrated flux fractions compared with the central "
        "plane-wave partition. Doubling the width reduces this finite-beam bias by more "
        "than threefold. The accepted packet uses the broader width. Both the direct "
        "continuum estimator audit and an independently injected negative branch scaling "
        "test preserve coefficient signs. The separate central Fourier estimator has a "
        "larger finite-window floor than the spatial template estimator; both are retained "
        "in JSON, and neither is silently substituted for the other."
    )
    paragraph("### Signed and complex coefficients")
    for key in ["P-200", "S-150", "reverse-P", "reverse-S"]:
        r = records[key]
        paragraph(f"**{key}: {'B→A' if r['reverse'] else 'A→B'}, h={r['h']:g} m, p=4.**")
        table(
            [
                "Branch",
                "analytic A",
                "Re(Ahat)",
                "Im(Ahat)",
                "abs(Ahat)",
                "phase residual (rad)",
                "complex error",
                "analytic flux",
                "inferred flux",
            ],
            [
                [
                    n,
                    f"{r['geometry']['reference']['branches'][n]['amplitude']:.8f}",
                    f"{b['amplitude']:.8f}",
                    f"{b['imaginary']:.3e}",
                    f"{b['magnitude']:.8f}",
                    f"{b['phase_residual']:.3e}",
                    f"{b['complex_error']:.3e}",
                    f"{r['geometry']['reference']['branches'][n]['flux']:.8f}",
                    f"{b['flux']:.8f}",
                ]
                for n, b in r["branches"].items()
            ],
        )
        paragraph(
            f"Outgoing inferred flux sum: {r['flux_sum']:.10f}; "
            f"closure residual: {r['closure_error']:.3e}. "
            f"Maximum relative complex error: {100 * maximum(r, 'relative_error'):.3f}%."
        )
    paragraph("### Refinement, phase angles and polarization")
    table(
        [
            "Case",
            "max signed-real error",
            "max complex error",
            "max flux-fraction error",
            "closure error",
            "max modal waveform error",
            "max polarization leakage",
            "max centroid discretization error (°)",
        ],
        [
            [
                k,
                *[
                    f"{maximum(records[k], q):.3e}"
                    for q in ["signed_error", "complex_error", "flux_error"]
                ],
                f"{records[k]['closure_error']:.3e}",
                f"{maximum(records[k], 'spectral_waveform_error'):.3e}",
                f"{maximum(records[k], 'polarization_leakage'):.3e}",
                f"{maximum(records[k], 'angle_discretization_error'):.3e}",
            ]
            for k in keys[:5]
        ],
    )
    paragraph(
        "The vector norms of signed, complex, flux and modal waveform errors decrease at "
        "each refinement; 300→200 m reduces them by more than 6.7 times in both primary "
        "cases. Individual very small coefficient errors need not decrease monotonically "
        "once temporal and estimator errors compete. No global exponential convergence "
        "claim is made across the material jump. The reverse-SV 200 m pilot failed the "
        "unchanged acceptance limits: its closure error was 1.3373e-2. Refining to 150 m "
        "reduced it to 6.6233e-4; the coarse record remains in the JSON."
    )
    table(
        [
            "Case/branch",
            "Snell (°)",
            "finite-continuum centroid (°)",
            "SEM centroid (°)",
            "SEM angular RMS (°)",
            "SEM−continuum absolute (°)",
            "polarization leakage",
            "arrival error (ms)",
        ],
        [
            [
                k + "/" + n,
                f"{r['geometry']['reference']['branches'][n]['angle']:.5f}",
                f"{b['continuum_angle']:.5f}",
                f"{b['angle']:.5f}",
                f"{b['spread']:.4f}",
                f"{b['angle_discretization_error']:.3e}",
                f"{b['polarization_leakage']:.3e}",
                f"{1000 * b['arrival_error']:.4f}"
                if b["arrival_error"] is not None
                else "unresolved",
            ]
            for k in ["P-200", "S-150", "reverse-P", "reverse-S"]
            for r in [records[k]]
            for n, b in r["branches"].items()
        ],
    )
    paragraph(
        "The finite-beam centroid can differ from the central Snell angle by about 1° "
        "for a weak converted branch. Its nonzero angular spread is reported rather than "
        "misinterpreted as numerical angle error. Arrival errors use the finite-packet "
        "envelope at independently placed ray receivers, with no fitted lag. All accepted "
        "principal-case envelope peaks lie inside the measurement windows."
    )
    paragraph(
        "![P field and predicted rays](2d_gll_sem_oblique_P.png)\n\n"
        "![SV field and predicted rays](2d_gll_sem_oblique_S.png)\n\n"
        "![Complex coefficient errors](2d_gll_sem_oblique_refinement.png)\n\n"
        "![Independent continuum receiver comparisons](2d_gll_sem_oblique_waveforms.png)"
    )
    paragraph("### Boundary isolation, controls and energy")
    table(
        [
            "Case",
            "left return (s)",
            "bottom (s)",
            "right (s)",
            "top (s)",
            "minimum margin (s)",
            "half-step energy relative range",
        ],
        [
            [
                k,
                *[
                    f"{r['geometry']['exterior_return_times'][s]:.6f}"
                    for s in ["left", "bottom", "right", "top"]
                ],
                f"{r['geometry']['return_margin']:.6f}",
                f"{r['energy_drift']:.3e}",
            ]
            for k in ["P-200", "S-150", "reverse-P", "reverse-S"]
            for r in [records[k]]
        ],
    )
    bd = records["boundary-control"]["differences"]
    paragraph(
        f"Moving all free walls from ±14400 to ±15600 m in the SV h=300 m control "
        f"changes receiver histories by {bd['history_relative']:.3e} relative and "
        f"any reported amplitude/imaginary/flux/angle scalar by at most "
        f"{max(v for k, v in bd.items() if k != 'history_relative'):.3e}. "
        "This checks actual boundary influence in addition to the conservative travel-time bound."
    )
    paragraph(
        "The oblique identical-material layered/homogeneous control is bitwise equal in "
        "both final fields, displacement/velocity histories and safe timestep. Its "
        "scattered reflected field (layered minus homogeneous) is exactly zero. This "
        "does not call the homogeneous packet's tiny incoming/tail content reflected energy."
    )
    paragraph("### Triangular comparison")
    table(
        [
            "Method",
            "h (m)",
            "DOFs",
            "max complex coefficient error",
            "max flux error",
            "closure error",
            "global projection residual",
        ],
        [
            [
                label,
                r["h"],
                r["dofs"],
                f"{maximum(r, 'complex_error'):.3e}",
                f"{maximum(r, 'flux_error'):.3e}",
                f"{r['closure_error']:.3e}",
                f"{r['projection_residual']:.3e}",
            ]
            for label, r in [("tri_p1", records["triangle-P"]), ("quad_gll p=4", records["P-200"])]
        ],
    )
    paragraph(
        "Both use the identical P packet, material pair, physical box, time step, "
        "analysis window and independent continuum diagnostics. The triangle run is "
        "illustrative, not an equal-work comparison or a new triangular convergence "
        "study. Its larger phase error is reported without treating it as truth. The "
        "existing isotropic oblique suite supplies the previous triangular refinement evidence."
    )
    paragraph("### MPI and compatibility")
    table(
        ["Diagnostic", "1 vs 2 ranks", "1 vs 4 ranks"],
        [
            [k, *[f"{records['mpi-' + str(n)]['errors'][k]:.3e}" for n in [2, 4]]]
            for k in [
                "mass",
                "action",
                "safe_dt",
                "material_audit",
                "damping",
                "trace",
                "final_u",
                "final_v",
            ]
        ],
    )
    table(
        ["Absolute derived difference", "1 vs 2 ranks", "1 vs 4 ranks"],
        [
            [
                label,
                *[
                    format(
                        max(
                            v
                            for k, v in records["mpi-" + str(n)]["errors"].items()
                            if k.endswith(suffix)
                        ),
                        ".3e",
                    )
                    for n in [2, 4]
                ],
            ]
            for label, suffix in [
                ("real amplitude", "_amplitude"),
                ("imaginary amplitude", "_imaginary"),
                ("flux", "_flux"),
                ("angle (°)", "_angle"),
            ]
        ],
    )
    paragraph(
        "The MPI partition-equivalence control uses p=4, h=600 m, with physical-coordinate "
        "ordering. Its coarse mesh is not an amplitude-accuracy case. Both materials are "
        "present on multiple ranks and the interface crosses partitions. All damping is "
        "zero. The stricter shared gate is 3e-11; measured field discrepancies are at "
        "floating-point accumulation scale."
    )
    if "owned-partitions-2" in records and "owned-partitions-4" in records:
        table(
            ["Ranks", "Owned cell counts (lower, upper), by rank; ghosts excluded"],
            [
                [
                    n,
                    "; ".join(
                        f"({r.get('0', 0)}, {r.get('1', 0)})"
                        for r in records[f"owned-partitions-{n}"]["owned_cell_counts_by_material"]
                    ),
                ]
                for n in [2, 4]
            ],
        )
    table(
        [
            "Archived 18be5ea compatibility",
            "arrays compared",
            "bitwise equal",
            "max absolute difference",
        ],
        [
            [
                k,
                len(v),
                all(a["bitwise"] for a in v.values()),
                f"{max(a['absolute_error'] for a in v.values()):.1e}",
            ]
            for k, v in records["compatibility"].items()
        ],
    )
    paragraph(
        "Production source files, sum-factorized volume forms, GLL quadrature, diagonal "
        "mass, timestep recurrence and public API are unchanged. This milestone adds "
        "validation utilities under tests only. No absorber is relied upon for any "
        "accepted interface measurement."
    )
    return "\n".join(lines).rstrip() + "\n"


def reference_audit():
    from tests.oblique2d.reference import Material
    from tests.oblique2d.reference import solve as old_solve

    from .reference import A, B, report

    errors = dict(
        continuity=0.0,
        flux_closure=0.0,
        old_amplitude=0.0,
        old_flux=0.0,
        horizontal_symmetry=0.0,
        normal_impedance=0.0,
    )
    for lower, upper in [(A, B), (B, A)]:
        for mode in ["P", "S"]:
            for angle in [0, 15, 25, -15]:
                r = report(mode, angle, lower, upper)
                old = old_solve(
                    mode,
                    angle,
                    lower=Material(lower.rho, lower.vp, lower.vs),
                    upper=Material(upper.rho, upper.vp, upper.vs),
                )
                errors["continuity"] = max(errors["continuity"], r["residual"])
                errors["flux_closure"] = max(errors["flux_closure"], r["closure"])
                opposite = report(mode, -angle, lower, upper)
                for n in NAMES:
                    b = r["branches"][n]
                    for quantity in ["amplitude", "flux"]:
                        errors["old_" + quantity] = max(
                            errors["old_" + quantity], abs(b[quantity] - old[n][quantity])
                        )
                    sign = 1 if n[1] == mode else -1
                    errors["horizontal_symmetry"] = max(
                        errors["horizontal_symmetry"],
                        abs(b["amplitude"] - sign * opposite["branches"][n]["amplitude"]),
                    )
                if angle == 0:
                    z1, z2 = lower.rho * lower.speed(mode), upper.rho * upper.speed(mode)
                    errors["normal_impedance"] = max(
                        errors["normal_impedance"],
                        abs(r["branches"]["R" + mode]["amplitude"] - (z2 - z1) / (z1 + z2)),
                        abs(r["branches"]["T" + mode]["amplitude"] - 2 * z1 / (z1 + z2)),
                    )
    return {k: float(v) for k, v in errors.items()}
