# ADR 0006: explicit coordinates and partitioned output

Status: accepted.

Context: seismic depth conventions can reverse force signs; wavefield gathers
and unstrided output cannot scale. Alternatives: implicit depth conversion,
a depth-only backend; rank-0 traces/fields; distributed HDF5 or ADIOS2 output.
Decision: positive-up z, explicitly encoded in YAML; no implicit depth transform.
DOLFINx interval geometry slot 0 carries z. Output uses collective XDMF/HDF5 for
P1 displacement snapshots and DG0 materials, plus one NPZ receiver shard per rank.
Reasons: current low-order fields are representable without order loss, the
installed HDF5 path is exercised under MPI, and Qt/ADIOS2 are not prerequisites.
Consequences: output stride is configurable, no snapshots by default, no global
wavefield gather. All ranks retain only their own receiver histories. Normalized
config and metadata are preserved; complete status is written after shards close.
Limitations: receiver histories are buffered, not streamed; one file per rank
will stress metadata servers at scale. Large runs need chunked distributed traces
and benchmarked VTX/ADIOS2 or another parallel format. Existing output directories
are rejected. No automatic restart or fault-tolerant MPI I/O is claimed.
