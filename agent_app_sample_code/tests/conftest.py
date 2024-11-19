import sys
import os

# Add the parent directory to sys.path, so that we can treat directories like
# `utils` as modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
