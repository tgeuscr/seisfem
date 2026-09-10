import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from dolfinx import io
from mpi4py import MPI
from typer.testing import CliRunner

from seisfem import Simulation, SimulationConfig
from seisfem.cli import app
from tests.helpers import pulse_config


def test_cli_api_output_and_snapshots(tmp_path):
    config = pulse_config(
        cells=200,
        time=dict(dt=0.002, duration=0.8),
        output=dict(
            directory=str(tmp_path / "run"),
            receiver_stride=3,
            snapshot_stride=100,
            energy_stride=20,
        ),
    )
    path = tmp_path / "model.yaml"
    config.to_yaml(path)
    runner = CliRunner()
    validated = runner.invoke(app, ["validate", str(path)])
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output)["valid"]
    completed = runner.invoke(app, ["run", str(path)])
    assert completed.exit_code == 0, completed.output
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["dolfinx"].startswith("0.11.")
    assert metadata["mesh"]["dofs"] == 201
    assert SimulationConfig.from_yaml(tmp_path / "run" / "config.yaml") == config
    data = config.model_dump(mode="json", by_alias=True)
    data["output"]["directory"] = None
    data["output"]["snapshot_stride"] = 0
    result = Simulation(SimulationConfig.model_validate(data)).run()
    with np.load(tmp_path / "run" / "receivers.rank00000.npz") as shard:
        for quantity in ("displacement", "velocity", "acceleration"):
            np.testing.assert_array_equal(shard[quantity], getattr(result, quantity))
        assert shard["time"][-1] == 0.8  # Include final step even off receiver stride.
    root = ET.parse(tmp_path / "run" / "wavefield.xdmf").getroot()
    times = [float(element.attrib["Value"]) for element in root.iter("Time")]
    np.testing.assert_allclose(times, [0, 0.2, 0.4, 0.6, 0.8])
    with io.XDMFFile(MPI.COMM_WORLD, tmp_path / "run" / "materials.xdmf", "r") as reader:
        mesh = reader.read_mesh()
        assert mesh.topology.index_map(1).size_global == 200
    # An existing run must never be silently clobbered.
    with pytest.raises(OSError, match="FileExistsError"):
        Simulation(config).run()


def test_requested_quantities_and_xarray():
    cfg = pulse_config(
        time=dict(dt=0.0004, duration=0.01),
        receivers=[dict(name="v_only", position=1400, quantities=["velocity"])],
    )
    result = Simulation(cfg).run()
    assert np.isnan(result.displacement).all()
    assert np.isfinite(result.velocity).all()
    assert result.requested.tolist() == [[False, True, False]]
    pytest.importorskip("xarray")
    ds = result.to_xarray()
    assert ds.velocity.dims == ("time", "receiver", "component")
    assert ds.velocity.attrs["units"] == "m s-1"


@pytest.mark.mpi
@pytest.mark.parametrize("ranks", [2, 4])
def test_mpi_matches_serial_and_single_source_insertion(tmp_path, ranks):
    # All these points lie on vertices, including interface/partition candidates.
    # Fewer receivers than ranks ensures empty shards in the 4-rank run.
    data = pulse_config(
        cells=400,
        layered=True,
        source=dict(position=2000, amplitude=8e6, frequency=10, time_shift=0.15),
        receivers=[dict(name="interface", position=2000), dict(name="left", position=1000)],
        time=dict(dt=0.001, duration=0.8),
        output=dict(directory=str(tmp_path / "mpi"), snapshot_stride=200, energy_stride=50),
    ).model_dump(mode="json", by_alias=True)
    cfg = SimulationConfig.model_validate(data)
    yaml_path = tmp_path / "mpi.yaml"
    cfg.to_yaml(yaml_path)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    command = ["mpiexec", "-n", str(ranks), sys.executable, "-m", "seisfem", "run", str(yaml_path)]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data["output"] = dict(energy_stride=50)
    serial = Simulation(SimulationConfig.model_validate(data)).run()
    count = np.zeros(2, dtype=int)
    for rank in range(ranks):
        with np.load(tmp_path / "mpi" / f"receivers.rank{rank:05d}.npz") as shard:
            ids = shard["receiver_ids"]
            count[ids] += 1
            for quantity in ("displacement", "velocity", "acceleration"):
                # Roundoff accumulates differently across MPI partitions. Acceleration
                # involves dt^-2 cancellation, so use a scale-relative absolute bound.
                reference = getattr(serial, quantity)[:, ids, :]
                scale = max(np.max(abs(reference), initial=0), 1)
                np.testing.assert_allclose(
                    shard[quantity], reference, rtol=2e-10, atol=2e-10 * scale
                )
            np.testing.assert_allclose(shard["energy"], serial.energy, rtol=1e-11, atol=1e-12)
    assert count.tolist() == [1, 1]
    metadata = json.loads((tmp_path / "mpi" / "metadata.json").read_text())
    assert metadata["mpi_size"] == ranks
    assert metadata["status"] == "complete"
    assert len(list(Path(tmp_path / "mpi").glob("receivers.*.npz"))) == ranks


def test_mesh_command_and_companion_guard(tmp_path):
    cfg = pulse_config(cells=100, time=dict(dt=0.001, duration=0.01))
    path = tmp_path / "mesh.yaml"
    cfg.to_yaml(path)
    destination = tmp_path / "mesh.xdmf"
    runner = CliRunner()
    result = runner.invoke(app, ["mesh", str(path), str(destination)])
    assert result.exit_code == 0, result.output
    assert destination.exists() and destination.with_suffix(".h5").exists()
    destination.unlink()  # The HDF5 companion alone must also prevent overwrite.
    result = runner.invoke(app, ["mesh", str(path), str(destination)])
    assert result.exit_code != 0
