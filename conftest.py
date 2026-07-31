"""
Root conftest.py for pytest configuration.
Ensures repository root is included in sys.path for clean package imports.
"""

import os
import sys

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
