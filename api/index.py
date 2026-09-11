"""
Vercel Serverless Entry Point for BioSentinel 2.0
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve directory paths for serverless execution
_API_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _API_DIR.parent
_CWD = Path.cwd()

# Ensure all candidate paths are in sys.path
for p in [str(_ROOT_DIR), str(_API_DIR), str(_CWD), str(_CWD.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from api.server import app
except (ImportError, ModuleNotFoundError):
    from server import app
