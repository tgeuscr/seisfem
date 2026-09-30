"""Collect compact evidence and produce quantitative field/error figures."""

import argparse
import json
from pathlib import Path

import numpy as np

from .packets import Packet
from .reference import NAMES, A, B, report
from .tables import quantitative, reference_audit


def figures(directory, destination, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in ["P", "S"]:
        h = 200 if mode == "P" else 150
        row = records[f"{mode}-{h}"]
        grid = np.load(directory / f"scattering-{mode}-{h}.npy")
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(
            np.linalg.norm(grid[:, :, :2], axis=-1),
            origin="lower",
            extent=[-7200, 7200, -7200, 7200],
            cmap="magma",
            vmin=0,
            vmax=0.6,
        )
        hit = np.array(row["geometry"]["hit"])
        ax.axhline(0, color="white", lw=0.8)
        for name, b in row["geometry"]["reference"]["branches"].items():
            center = np.array(b["center"])
            end = hit + 1.4 * (center - hit)
            ax.plot([hit[0], end[0]], [hit[1], end[1]], "c--", lw=0.8)
            ax.text(*center, name, color="cyan")
        ax.set(
            xlabel="x (m)",
            ylabel="z (m), positive up",
            title=f"{mode} incidence: displacement magnitude at {row['time']:.2f} s",
        )
        fig.colorbar(image, ax=ax, label="displacement magnitude (m)")
        fig.tight_layout()
        fig.savefig(destination / f"2d_gll_sem_oblique_{mode}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mode in zip(axes, ["P", "S"], strict=True):
        resolutions = [300, 200] if mode == "P" else [300, 200, 150]
        for name in NAMES:
            ax.loglog(
                resolutions,
                [records[f"{mode}-{h}"]["branches"][name]["complex_error"] for h in resolutions],
                "o-",
                label=name,
            )
        ax.set(
            xlabel="element width h (m)",
            ylabel="complex amplitude absolute error",
            title=f"{mode} incidence, p=4",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "2d_gll_sem_oblique_refinement.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for i, mode in enumerate(["P", "S"]):
        h = 200 if mode == "P" else 150
        raw = json.loads((directory / f"scattering-{mode}-{h}.json").read_text())
        packet = Packet(mode)
        times = np.asarray(raw["times"])
        keep = abs(times - raw["receiver_arrival"]) < 0.2
        times = times[keep]
        observed = np.asarray(raw["trace"])[keep]
        exact = packet.traces(raw["receiver_positions"], times)
        for j, name in enumerate(NAMES):
            d = np.array(raw["geometry"]["reference"]["branches"][name]["polarization"])
            axes[i, j].plot(times - raw["receiver_arrival"], observed[:, j, :2] @ d, label="SEM")
            axes[i, j].plot(
                times - raw["receiver_arrival"], exact[:, j, :2] @ d, "--", label="continuum packet"
            )
            axes[i, j].set(
                title=f"{mode} incidence, {name}",
                xlabel="time from ray arrival (s)",
                ylabel="projected u (m)",
            )
            axes[i, j].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "2d_gll_sem_oblique_waveforms.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = {
        p.stem.removeprefix("scattering-"): json.loads(p.read_text())
        for p in sorted(args.directory.glob("scattering-*.json"))
    }
    required = [
        "compatibility",
        "continuum-P",
        "continuum-S",
        "P-300",
        "P-200",
        "S-300",
        "S-200",
        "S-150",
        "reverse-P",
        "reverse-S",
        "boundary-control",
        "zero-contrast",
        "mpi-2",
        "mpi-4",
        "triangle-P",
        "reverse-S-coarse",
    ]
    assert set(required) <= records.keys(), set(required) - records.keys()
    # Keep histories sampled sparsely; raw fields and complete histories remain
    # in the run directory, not in the compact committed measurement artifact.
    for row in records.values():
        if "trace" in row:
            row["trace"] = row["trace"][::10]
            row["times"] = row["times"][::10]
            row["trace_decimation"] = 10
        row.pop("coordinate_audit", None)
        row.pop("material_audit", None)
    reference = {
        f"{mode}-{direction}": report(mode, 25 if mode == "P" else 15, lower, upper)
        for mode in ["P", "S"]
        for direction, lower, upper in [("AB", A, B), ("BA", B, A)]
    }
    output = dict(
        baseline="18be5ea031c42ecd6f138dfb536ef0bd771f5c04",
        production_changes=False,
        reference=reference,
        records=records,
        reference_audit=reference_audit(),
    )
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    figures(args.directory, args.output.parent, records)
    document = args.output.with_name("2d_gll_sem_oblique_interface.md")
    prefix = document.read_text().split("## Quantitative results")[0]
    document.write_text(prefix + quantitative(records, output["reference_audit"]))


if __name__ == "__main__":
    main()
