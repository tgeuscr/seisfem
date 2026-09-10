"""Serial/MPI frontend and numerical XDMF readback, via external HDF5 reader."""

import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from mpi4py import MPI
from typer.testing import CliRunner

from seisfem import Simulation, SimulationConfig
from seisfem.cli import app
from seisfem.output import runtime_metadata
from tests.audit.test_independent_numerics import config


def hdf_item(directory, item):
    """h5dump is an independent reader provided by the locked HDF5 package."""
    filename, dataset = item.text.strip().split(":", 1)
    output = subprocess.check_output(
        ["h5dump", "-d", dataset, "-y", "-w", "0", "-m", "%.17g", str(directory / filename)],
        text=True,
    )
    contents = re.search(r"DATA \{(.*?)\}", output, re.S).group(1)
    values = np.fromstring(contents.replace(",", " "), sep=" ")
    return values.reshape(tuple(int(i) for i in item.attrib["Dimensions"].split()))


def readback(directory):
    cfg = SimulationConfig.from_yaml(directory / "config.yaml")
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["config"] == cfg.model_dump(mode="json", by_alias=True)
    assert metadata["package_version"] == "0.1.0"
    assert metadata["petsc"] == [3, 25, 5]
    assert metadata["scalar_dtype"] == "float64"
    assert metadata["mesh"] == dict(cells=12, dofs=13, cell_type="interval", geometry_degree=1)
    traces = {q: np.empty((21, 13, 1)) for q in ["displacement", "velocity", "acceleration"]}
    counts = np.zeros(13, dtype=int)
    for filename in metadata["receiver_shards"]:
        with np.load(directory / filename) as shard:
            ids = shard["receiver_ids"]
            np.add.at(counts, ids, 1)
            assert shard["receiver_names"].tolist() == [f"node{i}" for i in ids]
            np.testing.assert_allclose(shard["positions"], np.linspace(-1, 1, 13)[ids], atol=1e-15)
            np.testing.assert_allclose(shard["time"], np.arange(21) * 0.001, atol=0, rtol=0)
            for quantity in traces:
                traces[quantity][:, ids] = shard[quantity]
    np.testing.assert_array_equal(counts, 1)
    root = ET.parse(directory / "materials.xdmf").getroot()
    z = hdf_item(directory, root.find(".//Geometry/DataItem"))[:, 0]
    cells = hdf_item(directory, root.find(".//Topology/DataItem")).astype(int)
    np.testing.assert_allclose(np.sort(z), np.linspace(-1, 1, 13), atol=2e-15)
    node_ids = np.rint((z + 1) * 6).astype(int)
    edges = np.sort(node_ids[cells], axis=1)
    edges = edges[np.argsort(edges[:, 0])]
    np.testing.assert_array_equal(edges, np.column_stack((np.arange(12), np.arange(1, 13))))
    midpoint = z[cells].mean(axis=1)
    rho = np.where(midpoint < 0, 1.3, 4.7)
    cp = np.where(midpoint < 0, 2.1, 3.2)
    # Expected DG0 values come directly from prescribed physical inputs.
    expected = dict(density=rho, mu=rho * (cp / 2) ** 2, propagation_modulus=rho * cp**2)
    expected["lambda"] = rho * (cp**2 - 2 * (cp / 2) ** 2)
    attributes = root.findall(".//Attribute")
    assert sorted(attribute.attrib["Name"] for attribute in attributes) == sorted(expected)
    for attribute in attributes:
        np.testing.assert_allclose(
            hdf_item(directory, attribute.find("DataItem")).ravel(),
            expected[attribute.attrib["Name"]],
            rtol=2e-14,
        )
    root = ET.parse(directory / "wavefield.xdmf").getroot()
    field_z = hdf_item(directory, root.find(".//Geometry/DataItem"))[:, 0]
    node_ids = np.rint((field_z + 1) * 6).astype(int)
    times = []
    for grid in root.findall(".//Grid[@GridType='Uniform']"):
        attribute = grid.find("Attribute")
        if attribute is None:
            continue
        time = float(grid.find("Time").attrib["Value"])
        times.append(time)
        data = hdf_item(directory, attribute.find("DataItem")).ravel()
        np.testing.assert_allclose(
            data, traces["displacement"][round(time / 0.001), node_ids, 0], atol=1e-15, rtol=1e-13
        )
    np.testing.assert_allclose(times, [0, 0.007, 0.014, 0.020], atol=1e-16)
    assert np.max(abs(traces["displacement"][-1])) > 1e-5
    return cfg, traces


@pytest.mark.mpi
@pytest.mark.parametrize("ranks", [1, 2, 3, 4])
def test_actual_partition_edges_and_field_readback(tmp_path, ranks):
    assert shutil.which("h5dump"), "Locked HDF5 reader is required for independent value readback"
    directory = tmp_path / "distributed"
    result = subprocess.run(
        [
            "mpiexec",
            "-n",
            str(ranks),
            sys.executable,
            "-m",
            "tests.audit.mpi_probe",
            str(directory),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cfg, traces = readback(directory)
    summary = json.loads((directory / "probe.json").read_text())
    print("actual partitions", ranks, summary["shared_vertices"])
    if ranks > 1:
        assert summary["shared_vertices"]
    # Three frontends: direct config, YAML/Python, and YAML/CLI.
    data = cfg.model_dump(mode="json", by_alias=True)
    data["output"] = dict(energy_stride=3)
    direct_cfg = SimulationConfig.model_validate(data)
    direct = Simulation(direct_cfg, comm=MPI.COMM_SELF).run()
    path = tmp_path / "serial.yaml"
    direct_cfg.to_yaml(path)
    yaml_result = Simulation(SimulationConfig.from_yaml(path), comm=MPI.COMM_SELF).run()
    data["output"] = dict(directory=str(tmp_path / "cli"), energy_stride=3)
    cli_cfg = SimulationConfig.model_validate(data)
    cli_cfg.to_yaml(path)
    completed = CliRunner().invoke(app, ["run", str(path)])
    assert completed.exit_code == 0, completed.output
    inspected = CliRunner().invoke(app, ["inspect", str(path)])
    assert json.loads(inspected.output)["config"] == cli_cfg.model_dump(mode="json", by_alias=True)
    with np.load(tmp_path / "cli" / "receivers.rank00000.npz") as cli:
        for quantity in traces:
            np.testing.assert_array_equal(getattr(direct, quantity), getattr(yaml_result, quantity))
            np.testing.assert_array_equal(cli[quantity], getattr(direct, quantity))
            np.testing.assert_allclose(
                traces[quantity], getattr(direct, quantity), rtol=3e-12, atol=2e-12
            )


def test_interrupted_output_is_not_complete_and_cannot_be_overwritten(tmp_path):
    data = config().model_dump(mode="json", by_alias=True)
    destination = tmp_path / "interrupted"
    data["output"] = dict(directory=str(destination), snapshot_stride=1)
    cfg = SimulationConfig.model_validate(data)

    def interrupt(event):
        if event.step == 2:
            raise RuntimeError("audit interruption")

    sim = Simulation(cfg, comm=MPI.COMM_SELF)
    with pytest.raises(RuntimeError, match="audit interruption"):
        sim.run(progress=interrupt)
    assert sim.operators is None
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["status"] == "running"
    assert not list(destination.glob("receivers.*.npz"))
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(OSError, match="FileExistsError"):
        Simulation(cfg, comm=MPI.COMM_SELF).run()
    assert before == {p.name: p.read_bytes() for p in destination.iterdir()}


def test_source_identity_clean_dirty_detached_and_unavailable(tmp_path, monkeypatch):
    # Use an isolated miniature checkout; never change the actual baseline HEAD.
    from seisfem import output

    root = tmp_path / "checkout"
    package = root / "src" / "seisfem"
    package.mkdir(parents=True)
    module = package / "output.py"
    module.write_text("# audit identity fixture\n")
    monkeypatch.setattr(output, "__file__", str(module))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Audit Fixture",
        "-c",
        "user.email=audit@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    cfg = config()
    clean = runtime_metadata(cfg, MPI.COMM_SELF)["source_identity"]
    assert clean["git_commit"] == git("rev-parse", "HEAD") and clean["git_dirty"] is False
    git("checkout", "--detach", "-q", "HEAD")
    assert runtime_metadata(cfg, MPI.COMM_SELF)["source_identity"] == clean
    module.write_text("# changed audit identity fixture\n")
    dirty = runtime_metadata(cfg, MPI.COMM_SELF)["source_identity"]
    assert dirty["git_dirty"] is True and dirty["git_commit"] == clean["git_commit"]
    assert dirty["source_sha256"] != clean["source_sha256"]
    shutil.rmtree(root / ".git")
    unavailable = runtime_metadata(cfg, MPI.COMM_SELF)["source_identity"]
    assert unavailable["git_commit"] is None
    # Existing behavior: false dirty state here means unavailable, not proven clean.
    assert unavailable["git_dirty"] is False
