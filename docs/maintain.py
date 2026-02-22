#!/usr/bin/env python3
# Project path: docs/maintain.py
"""
Run maintenance utilities with consistent, minimal output:
- project_structure/project_structure.py
- project_analytics/project_analytics.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import contextmanager

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
PROJECT_STRUCTURE_DIR = DOCS_DIR / "project_structure"
PROJECT_ANALYTICS_DIR = DOCS_DIR / "project_analytics"


@contextmanager
def temp_argv(argv):
    orig = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = orig


def run_project_structure():
    sys.path.insert(0, str(PROJECT_STRUCTURE_DIR))
    import project_structure  # type: ignore

    with temp_argv([str(project_structure.__file__)]):
        project_structure.main()
    return f"structure: wrote {PROJECT_STRUCTURE_DIR / 'project_structure.txt'}"


def run_project_analytics():
    sys.path.insert(0, str(PROJECT_ANALYTICS_DIR))
    import project_analytics  # type: ignore

    os.environ["PROJECT_ANALYTICS_QUIET"] = "1"
    with temp_argv([str(project_analytics.__file__)]):
        project_analytics.main()
    return f"analytics: wrote {PROJECT_ANALYTICS_DIR / 'project_analytics.txt'}"


def main():
    results = []
    results.append(run_project_structure())
    results.append(run_project_analytics())
    for line in results:
        print(line)


if __name__ == "__main__":
    main()
