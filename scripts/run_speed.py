"""Thin wrapper: python scripts/run_speed.py --config configs/experiment/exp_fp8.yaml"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "speed"] + sys.argv[1:]
    main()
