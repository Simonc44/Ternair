"""Make ``python -m ternair`` work by delegating to the CLI."""
from ternair.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
