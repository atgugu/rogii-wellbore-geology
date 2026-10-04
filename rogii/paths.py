"""Where data is read from and results are written.

Competition data is not redistributed here. Download it (see the README) and
either place it in ``data/`` or point ``ROGII_DATA`` at it.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("ROGII_DATA", os.path.join(ROOT, "data"))
RESULTS_DIR = os.path.join(ROOT, "results")
