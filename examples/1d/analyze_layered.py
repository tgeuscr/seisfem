"""Offline analytical comparison/plot of the supplied layered P experiment.

Run after `seisfem run examples/1d/layered.yaml`. Reads receiver shards only;
this postprocessing program does not implement a numerical solver.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seisfem.config import Layered, SimulationConfig
from seisfem.sources import ricker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path("outputs/layered"))
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()
    cfg = SimulationConfig.from_yaml(args.directory / "config.yaml")
    if not isinstance(cfg.materials, Layered) or len(cfg.materials.layers) != 2:
        raise ValueError("This analytical comparison requires exactly two layers")
    source = cfg.source
    first, second = cfg.materials.layers
    interface = first.upper
    if source is None or not source.position < interface:
        raise ValueError("Reference assumes an interior point force below the interface")
    c1, c2 = first.material.speed(cfg.mode), second.material.speed(cfg.mode)
    z1, z2 = first.material.density * c1, second.material.density * c2
    reflection, transmission = (z1 - z2) / (z1 + z2), 2 * z1 / (z1 + z2)
    incident_amplitude = source.amplitude / (2 * z1)
    records = {}
    for path in args.directory.glob("receivers.rank*.npz"):
        with np.load(path) as shard:
            for j, receiver_id in enumerate(shard["receiver_ids"]):
                records[int(receiver_id)] = (
                    shard["time"].copy(),
                    shard["velocity"][:, j, 0].copy(),
                )
    fig, axes = plt.subplots(len(cfg.receivers), 1, figsize=(9, 6), sharex=True, squeeze=False)
    measurements = []
    for i, receiver in enumerate(cfg.receivers):
        time, values = records[i]
        if receiver.position < interface:
            arrivals = [
                ("direct", abs(receiver.position - source.position) / c1, incident_amplitude),
                (
                    "reflected",
                    (2 * interface - source.position - receiver.position) / c1,
                    reflection * incident_amplitude,
                ),
            ]
        else:
            arrivals = [
                (
                    "transmitted",
                    (interface - source.position) / c1 + (receiver.position - interface) / c2,
                    transmission * incident_amplitude,
                )
            ]
        reference = np.zeros_like(time)
        ax = axes[i, 0]
        for name, travel, amplitude in arrivals:
            arrival = source.time_shift + travel
            causal = time >= travel
            reference[causal] += amplitude * ricker(time[causal], source.frequency, arrival)
            window = abs(time - arrival) < 0.07
            peak = np.argmax(np.sign(amplitude) * values[window])
            measurements.append(
                {
                    "receiver": receiver.name,
                    "phase": name,
                    "predicted_peak_s": arrival,
                    "measured_peak_s": time[window][peak],
                    "predicted_velocity_m_s": amplitude,
                    "measured_velocity_m_s": values[window][peak],
                }
            )
            ax.axvline(arrival, color="0.6", linewidth=0.8)
            ax.text(arrival + 0.01, 0.94, name, transform=ax.get_xaxis_transform(), va="top")
        ax.plot(time, values, color="#116466", label="DOLFINx P1")
        ax.plot(time, reference, "--", color="#bc572a", label="Infinite-layer analytical reference")
        ax.set_ylabel("Velocity [m/s]")
        ax.set_title(f"{receiver.name}: z = {receiver.position:g} m", loc="left")
        ax.grid(alpha=0.15)
        ax.legend(loc="lower left", fontsize=8)
    axes[-1, 0].set_xlabel("Time [s]")
    fig.suptitle(f"Layered {cfg.mode} propagation · R = {reflection:.6f}, T = {transmission:.6f}")
    fig.tight_layout()
    figure = args.figure or args.directory / "seismograms.svg"
    fig.savefig(figure)
    plt.close(fig)
    report = {
        "reflection": reflection,
        "transmission": transmission,
        "measurements": measurements,
        "note": "Analytical curves omit finite-domain boundary returns.",
    }
    (args.directory / "analytical_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
