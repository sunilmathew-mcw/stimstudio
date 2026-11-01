"""Allow the application to run with ``python -m stimulus_forge``."""

import sys

if "--cli" in sys.argv:
    from stimulus_forge.cli import run_cli
    sys.argv.remove("--cli")
    raise SystemExit(run_cli(sys.argv[1:]))

from stimulus_forge.app import main


if __name__ == "__main__":
    raise SystemExit(main())
