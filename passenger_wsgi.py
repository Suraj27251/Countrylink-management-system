"""WSGI entrypoint for Passenger.

Avoid importing this file via imp.load_source, which can recurse indefinitely
if the source path also points to passenger_wsgi.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Passenger looks for `application` by default.
from app import app as application  # noqa: E402
