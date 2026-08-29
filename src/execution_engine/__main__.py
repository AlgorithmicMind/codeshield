"""Entry point for `python -m execution_engine`."""

from __future__ import annotations

import sys

from execution_engine.cli import main

if __name__ == "__main__":
    sys.exit(main())
