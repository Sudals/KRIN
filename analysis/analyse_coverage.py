"""Entry point for run-paired coverage statistics and crossed bootstrap intervals.

Use --verify-baselines to audit scoring support against the processed datasets.
See recalculate_coverage.py --help for explicit matched-rerun inputs.
"""
from recalculate_coverage import main

if __name__ == "__main__":
    main()
