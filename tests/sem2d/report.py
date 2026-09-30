"""Collect measured SEM results and append reproducible tables/plots to the report.

python -m tests.sem2d.report /tmp/sem-final docs/validation/2d_gll_sem_measurements.json
"""

import argparse
import json
import platform
from pathlib import Path


def label(p):
    return "tri P1" if p == 0 else f"GLL p={p}"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def number(x):
    return f"{x:.5g}"


def collect(directory):
    data = {}
    omitted = {"trace", "velocity_trace", "field_samples", "coordinate_audit"}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        data[path.stem] = {k: v for k, v in record.items() if k not in omitted}
    required = [f"operators-{p}" for p in range(1, 7)]
    required += ["mpi-2", "mpi-4", "boundary-P", "boundary-S"]
    required += [
        f"benchmark-p{p}-n{n}-{mode}-f0.02"
        for p in [0, 2, 4, 6]
        for n in [48, 96, 144]
        for mode in ["P", "S"]
    ]
    missing = set(required) - data.keys()
    if missing:
        raise ValueError(f"Missing measurements: {sorted(missing)}")
    return data


def result_tables(data):
    parts = []

    def section(title, headers, rows):
        parts.extend([f"### {title}", table(headers, rows)])

    section(
        "Independent operator checks (rho=2, domain area=1.5)",
        [
            "p",
            "DOFs",
            "mass offdiag abs/rel",
            "mass error/component",
            "K symmetry",
            "rigid",
            "K reference",
        ],
        [
            [
                p,
                r["dofs"],
                number(r["mass_offdiagonal_absolute"])
                + " / "
                + number(r["mass_offdiagonal_relative"]),
                number(max(abs(v - 3) for v in r["mass_total"])),
            ]
            + [number(r[k]) for k in ["symmetry", "rigid_residual", "reference_K_error"]]
            for p in range(1, 7)
            for r in [data[f"operators-{p}"]]
        ],
    )
    section(
        "Actual eigenvalue stability and energy (h=0.5, Vp=3, Vs=1.5)",
        [
            "p",
            "DOFs",
            "dtcrit",
            "safe bound",
            "max norm at .95",
            "max norm at 1.01",
            "energy drift",
        ],
        [
            [p, r["dofs"]]
            + [
                number(r[k])
                for k in [
                    "dtcrit",
                    "safe_bound",
                    "stable_max",
                    "unstable_max",
                    "conserved_energy_drift",
                ]
            ]
            for p in [1, 2, 4, 6]
            for r in [data[f"stability-{p}"]]
        ],
    )
    section(
        "Spatial refinement at 25 degrees, dt/dtcrit about 0.02",
        [
            "mode",
            "method",
            "DOFs",
            "h element m",
            "PPW",
            "dtcrit s",
            "speed error %",
            "field L2 %",
            "trace L2 %",
        ],
        [
            [mode, label(p), r["dofs"]]
            + [number(r[k]) for k in ["h_element", "points_per_wavelength", "dtcrit"]]
            + [number(100 * r[k]) for k in ["speed_error", "field_error", "waveform_error"]]
            for mode in ["P", "S"]
            for p in [0, 2, 4, 6]
            for n in [48, 96, 144]
            for r in [data[f"benchmark-p{p}-n{n}-{mode}-f0.02"]]
        ],
    )
    section(
        "Equal DOFs: 18,818, temporal fraction 0.02",
        [
            "mode",
            "method",
            "steps",
            "DOF steps",
            "assembly s",
            "eigen s",
            "step s",
            "total s",
            "polarization",
            "arrival error %",
        ],
        [
            [mode, label(p), r["steps"], r["work_proxy"]]
            + [
                number(r[k])
                for k in [
                    "assembly_seconds",
                    "eigensolve_seconds",
                    "stepping_seconds",
                    "backend_seconds",
                    "polarization_error",
                ]
            ]
            + [number(100 * r["arrival_error"])]
            for mode in ["P", "S"]
            for p in [0, 2, 4, 6]
            for r in [data[f"benchmark-p{p}-n96-{mode}-f0.02"]]
        ],
    )
    section(
        "Order study: 12 by 12 quadrilateral elements",
        [
            "mode",
            "p",
            "DOFs",
            "PPW",
            "dtcrit s",
            "speed error %",
            "field L2 %",
            "DOF steps",
            "total s",
        ],
        [
            [mode, p, r["dofs"], number(r["points_per_wavelength"]), number(r["dtcrit"])]
            + [
                number(100 * r["speed_error"]) if r["points_per_wavelength"] >= 2 else "unresolved",
                number(100 * r["field_error"]),
            ]
            + [r["work_proxy"], number(r["backend_seconds"])]
            for mode in ["P", "S"]
            for p in [2, 4, 6]
            for r in [data[f"order-p{p}-{mode}"]]
        ],
    )
    section(
        "Signed phase-speed error (%) versus requested dt/dtcrit at 18,818 DOFs",
        ["mode", "method", ".02", ".1", ".2", ".4", ".6", ".8"],
        [
            [mode, label(p)]
            + [
                number(100 * data[f"benchmark-p{p}-n96-{mode}-f{f}"]["signed_speed_error"])
                for f in [0.02, 0.1, 0.2, 0.4, 0.6, 0.8]
            ]
            for mode in ["P", "S"]
            for p in [0, 2, 4, 6]
        ],
    )
    candidates = [r for name, r in data.items() if name.startswith(("benchmark-", "accuracy-"))]
    rows = []
    public_rows = []
    for error in ["field_error", "waveform_error"]:
        for target in [0.01, 0.001]:
            for p in [0, 2, 4, 6]:
                choices = []
                for a in candidates:
                    if a["degree"] != p or a["branch"] != "P":
                        continue
                    matches = [
                        b
                        for b in candidates
                        if b["degree"] == p
                        and b["branch"] == "S"
                        and b["effective"] == a["effective"]
                        and b["requested_fraction"] == a["requested_fraction"]
                    ]
                    for b in matches:
                        if max(a[error], b[error]) <= target:
                            choices.append((a["backend_seconds"] + b["backend_seconds"], a, b))
                if error == "waveform_error":
                    eligible = [
                        entry
                        for entry in choices
                        if all(r["dt"] <= 0.9 * r["safe_dt"] for r in entry[1:])
                    ]
                    if eligible:
                        seconds, a, b = min(eligible, key=lambda x: x[0])
                        public_rows.append(
                            [
                                100 * target,
                                label(p),
                                a["dofs"],
                                a["requested_fraction"],
                                number(100 * max(a[error], b[error])),
                                a["work_proxy"] + b["work_proxy"],
                                number(seconds),
                            ]
                        )
                    else:
                        public_rows.append(
                            [100 * target, label(p), "not attained", "—", "—", "—", "—"]
                        )
                if choices:
                    seconds, a, b = min(choices, key=lambda x: x[0])
                    rows.append(
                        [
                            error.replace("_error", ""),
                            100 * target,
                            label(p),
                            a["dofs"],
                            a["requested_fraction"],
                            number(100 * max(a[error], b[error])),
                            a["work_proxy"] + b["work_proxy"],
                            number(seconds),
                        ]
                    )
                else:
                    rows.append(
                        [
                            error.replace("_error", ""),
                            100 * target,
                            label(p),
                            "not attained",
                            "—",
                            "—",
                            "—",
                            "—",
                        ]
                    )
    section(
        "Fixed accuracy: fastest tested pair satisfying BOTH P and SV",
        [
            "metric",
            "target %",
            "method",
            "DOFs/run",
            "dt fraction",
            "worst error %",
            "pair DOF steps",
            "pair total s",
        ],
        rows,
    )
    section(
        "Waveform targets restricted to the default public timestep bound",
        [
            "target %",
            "method",
            "DOFs/run",
            "dt fraction",
            "worst error %",
            "pair DOF steps",
            "pair total s",
        ],
        public_rows,
    )
    section(
        "Axial, diagonal, negative-angle validation",
        [
            "mode",
            "p",
            "angle deg",
            "DOFs",
            "speed error %",
            "polarization",
            "arrival error %",
            "phase-angle error deg",
            "field L2 %",
        ],
        [
            [
                r["branch"],
                r["degree"],
                r["angle"],
                r["dofs"],
                number(100 * r["speed_error"]),
                number(r["polarization_error"]),
                number(100 * r["arrival_error"]),
                number(r["angle_error"]),
                number(100 * r["field_error"]),
            ]
            for name, r in data.items()
            if name.startswith("angle-")
        ],
    )
    section(
        "Boundary displacement control: absolute change in diagnostic",
        ["mode", "field error", "speed error", "trace error", "angle error deg"],
        [
            [mode]
            + [
                number(data[f"boundary-{mode}"][k])
                for k in ["field_error", "speed_error", "waveform_error", "angle_error"]
            ]
            for mode in ["P", "S"]
        ],
    )
    section(
        "MPI differences relative to serial",
        ["diagnostic", "2 ranks abs", "2 ranks relative", "4 ranks abs", "4 ranks relative"],
        [
            [k]
            + [
                number(data[f"mpi-{rank}"][k][measure])
                for rank in [2, 4]
                for measure in ["absolute", "relative"]
            ]
            for k in [
                "diagonal_mass",
                "stiffness_action",
                "final_u",
                "final_v",
                ".packet.dtcrit",
                ".packet.trace",
                ".packet.velocity_trace",
                ".packet.field_error",
                ".packet.phase_angle",
                ".displacement",
                ".velocity",
            ]
        ],
    )
    parts.append(
        "Relative MPI errors use each physical field's maximum scale; near-zero residual "
        "ratios are not accuracy metrics. Source entries refer to the separate public "
        "point-force run."
    )
    section(
        "Continuum audit and temporal order",
        ["check", "values"],
        [
            [
                f"{mode} initial potential / spectral resolution",
                " / ".join(number(v) for v in data[f"continuum-{mode}"].values()),
            ]
            for mode in ["P", "S"]
        ]
        + [
            [
                f"p={p} observed temporal orders",
                ", ".join(number(v) for v in data[f"temporal-exact-{p}"]["rates"]),
            ]
            for p in [2, 4, 6]
        ],
    )
    packet_rows = [r for r in data.values() if "energy_drift" in r]
    parts.append(
        "Maximum packet half-step energy drift across recorded runs: "
        f"{max(r['energy_drift'] for r in packet_rows):.5g}. Machine-readable records retain "
        "all errors, times, angles, resolutions, and timestep fractions; raw receiver "
        "arrays are omitted from the compact audit file."
    )
    return "\n\n".join(parts) + "\n"


def plots(data, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, mode in enumerate(["P", "S"]):
        for p in [0, 2, 4, 6]:
            rows = [data[f"benchmark-p{p}-n{n}-{mode}-f0.02"] for n in [48, 96, 144]]
            axes[i, 0].loglog(
                [r["dofs"] for r in rows], [r["field_error"] for r in rows], "o-", label=label(p)
            )
            axes[i, 1].loglog(
                [r["backend_seconds"] for r in rows],
                [r["field_error"] for r in rows],
                "o-",
                label=label(p),
            )
            rows = [
                data[f"benchmark-p{p}-n96-{mode}-f{f}"] for f in [0.02, 0.1, 0.2, 0.4, 0.6, 0.8]
            ]
            axes[i, 2].plot(
                [r["actual_fraction"] for r in rows],
                [100 * r["signed_speed_error"] for r in rows],
                "o-",
                label=label(p),
            )
        for j, xlabel in enumerate(
            ["Global DOFs", "Assembly + eigen solve + stepping (s)", "dt / dtcrit"]
        ):
            axes[i, j].set_xlabel(xlabel)
            axes[i, j].set_ylabel(
                f"{mode}: " + ("relative field L2 error" if j < 2 else "signed speed error (%)")
            )
            axes[i, j].grid(True, alpha=0.3)
        axes[i, 2].axhline(0, color="k", lw=0.7)
    axes[0, 0].legend()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = collect(args.directory)
    payload = dict(
        provenance=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            baseline="b4df73b58463415971093af84c01addd0cdb3f6d",
            threads=1,
            timing="single observations, warm JIT cache; see report",
        ),
        measurements=data,
    )
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    report = args.output.with_name("2d_gll_sem.md")
    prefix = report.read_text().split("\n## Measured results\n")[0]
    report.write_text(prefix.rstrip() + "\n\n## Measured results\n\n" + result_tables(data))
    plots(data, args.output.with_name("2d_gll_sem_comparison.png"))


if __name__ == "__main__":
    main()
