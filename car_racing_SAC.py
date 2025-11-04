"""Legacy SAC launcher kept for compatibility. Delegates to the unified train CLI."""

import sys

from train import main


if __name__ == "__main__":
    if "--algo" not in sys.argv:
        sys.argv.extend(["--algo", "sac"])
    main()

