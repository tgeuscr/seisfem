# Supported environment and reproducibility

The verified binary stack is Linux x86-64, Python 3.14.6, DOLFINx/Basix 0.11.0,
UFL 2026.1.0, real-double PETSc 3.25.5 and MPICH 5.0.1. These are installed and
executed, not inferred from dependency declarations. DOLFINx and MPI are supplied
by conda-forge; pip installs only the project and Python-level dependencies.

`environment.yml` is a maintainable environment intent. The explicit conda lock
records the tested build URLs and MD5 artifact checksums; `pip-lock.txt` supplies
exact versions of packages added via pip. The lock was formed from the installed
transitive dependency closure of the backend and verification tools, excluding
the original interactive environment's Qt/Jupyter/VTK stack. Headless
matplotlib-base/pandas are included for the figure/Xarray checks. Package extras
keep these optional for a minimal backend install.

The lock contains a conda x86-64 microarchitecture level-3 requirement. Use it on
a compatible CPU; re-solve `environment.yml` for older CPUs, other OS/architecture,
or site-specific MPI. A different binary stack must rerun the verification suite.
The current run does not establish multi-node fabric performance. Never mix a
system mpiexec with an incompatible conda mpi4py/PETSc build. Activate the same
environment for both the launcher and Python processes.

The reproducibility manifest preserves normalized input, source-file SHA256,
Git commit/dirty state if available, versions, MPI library/size, mesh/DOF count,
scalar type and assembled stability bound. An uncommitted new repository has no
Git commit; it is recorded as null, not fabricated. The source digest identifies
code but cannot replace archiving that code. Archive the checkout, locks, config
and result directory together for a scientific record.

When Git is unavailable, `git_commit` is null and the current metadata reports
`git_dirty: false`; that false value is not evidence of a clean checkout. Interpret
the dirty flag only when Git identity is available. A detached HEAD still records
its commit normally. The Python source digest does not cover dependency binaries.

No initial-condition files, imported meshes or external material assets are used
in this milestone. A complete config therefore specifies all model inputs. Source
code updates and changed dependency builds can still change numerical results.
