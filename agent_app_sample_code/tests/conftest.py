import sys
import os

# Add the root directory to sys.path, so that we can treat directories like
# agent_app_sample_code as modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
