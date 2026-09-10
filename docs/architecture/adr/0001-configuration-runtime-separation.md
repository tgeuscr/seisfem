# ADR 0001: immutable configuration, composed runtime

Status: accepted, first milestone.

Context: CLI, notebooks and future Qt must describe exactly the same experiment,
while MPI communicators and FE objects cannot be serialized as configuration.
Alternatives: mutable dataclasses with manual checking; runtime-rich configuration;
Pydantic values plus independent runtime resources.
Decision: frozen Pydantic v2 models with unknown-key and nonfinite-value rejection,
YAML input, explicit defaults, and immutable nested tuples. Runtime composition
uses a small Simulation lifecycle and inspectable spatial operators.
Reasons: cross-field physical checks, schema support, normalized serialization,
and no duplicate validation/physics in frontends.
Consequences: parsing depends on Pydantic/PyYAML, but does not import DOLFINx. MPI,
PETSc and Functions exist only in runtime modules. CLI/Qt remain optional extras.
Limitations: only schema version 1 exists; migrations are required when scientific
meaning changes. Initial data are zero in declarative runs; runtime integrator
initial data exist for verification, not as a hidden reproducible frontend option.
