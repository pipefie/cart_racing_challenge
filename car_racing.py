"""Legacy entrypoint retained for backward compatibility.

Prefer running ``python train.py`` with explicit CLI arguments. This module simply
delegates to ``train.main`` so existing automation keeps working.
"""

from train import main

if __name__ == "__main__":
    main()
