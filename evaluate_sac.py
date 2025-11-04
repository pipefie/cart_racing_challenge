"""Legacy evaluation script that routes to the new evaluate.py CLI."""

import sys

from evaluate import main

if __name__ == "__main__":
    if "--algo" not in sys.argv:
        sys.argv.extend(["--algo", "sac"])
    main()
