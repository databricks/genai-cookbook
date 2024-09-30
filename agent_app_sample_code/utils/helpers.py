import subprocess
import sys

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        # dbutils.library.installPyPI(package)
        %pip install package
        dbutils.library.restartPython()
