"""Entry point for the aligned ten-experiment suite."""

from __future__ import annotations

if __package__:
    from .experiments import main
else:
    import sys
    from pathlib import Path
    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from rtt_thesis_audited_aligned.experiments import main

if __name__ == "__main__":
    main()
