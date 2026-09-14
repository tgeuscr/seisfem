"""Small collective input guard for the 2D experiment layer."""


def collective_input(comm, factory, label):
    """Finish rank-local validation everywhere before any subsequent collective.

    factory must not call collectives. All participating ranks must call this
    guard, including ranks with invalid input.
    """
    value, error = None, None
    try:
        value = factory()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = comm.allgather(error)
    if any(error is not None for error in errors):
        failures = "; ".join(f"rank {i}: {error}" for i, error in enumerate(errors) if error)
        raise ValueError(f"Invalid {label}; {failures}")
    return value


def identical_input(comm, signature, label):
    """Reject differing replicated inputs on every participating rank."""
    signatures = comm.allgather(signature)
    if any(value != signatures[0] for value in signatures):
        raise ValueError(f"All ranks must use identical {label}")
