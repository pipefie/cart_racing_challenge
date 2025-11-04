"""Deprecated filename kept to avoid breaking shortcuts. Calls evaluate.py."""

import sys

from evaluate import main

if __name__ == "__main__":
    if "--algo" not in sys.argv:
        sys.argv.extend(["--algo", "ppo"])
    main()
