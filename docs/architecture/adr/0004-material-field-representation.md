# ADR 0004: aligned cellwise isotropic properties

Status: accepted.

Context: geological jumps must retain their prescribed positions and flux physics.
Alternatives: continuous nodal interpolation, midpoint sampling on arbitrary cells,
cut-cell integration, aligned DG0 fields.
Decision: require uniform-mesh vertices at every layer interface; assign DG0 rho,
lambda and mu. Material mapping is separate from the P/S constitutive reduction.
Reasons: no smearing or mesh-dependent interface shifts; exact constant cell
integrals; the same property conversion will feed future vector forms.
Consequences: reject gaps, overlaps, incomplete coverage and misaligned interfaces.
Direct Lamé input is supported; positive bulk and shear moduli enforce solid
strain-energy positivity while allowing negative lambda.
Limitations: interfaces sorted in increasing positive-up z, no fluid layers,
nonuniform meshes or arbitrary heterogeneous input. Meshtags/imported models will
replace spatial mapping, not the constitutive law, in later milestones.
