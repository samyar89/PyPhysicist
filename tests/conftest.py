import os
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_PARENT_DIR = os.path.abspath(os.path.join(REPO_DIR, '..'))

for path in (REPO_DIR, REPO_PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
