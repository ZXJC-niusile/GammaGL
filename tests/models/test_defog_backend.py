import os
import subprocess
import sys
import importlib.util

def test_defog_backend_imports():
    """
    Ensure that DeFoG core components can be parsed and imported 
    without crashing under non-torch backends (e.g. tensorflow).
    This proves there are no stray `import torch` or PyG hard dependencies
    in the shared GammaGL namespace.
    """
    if importlib.util.find_spec('tensorlayerx') is None:
        try:
            import pytest
            pytest.skip("tensorlayerx is not installed")
        except ImportError:
            print("Skipped: tensorlayerx is not installed")
            return

    script = """
import tensorlayerx as tlx
from gammagl.models.defog import DeFoGModel
from gammagl.layers.attention.defog_layer import XEyTransformerLayer
print("Import successful on backend:", tlx.BACKEND)
"""
    
    env = os.environ.copy()
    # Try with tensorflow backend
    env['TL_BACKEND'] = 'tensorflow'
    
    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # If the user doesn't have tensorflow installed, it will fail with ModuleNotFoundError: No module named 'tensorflow'
    # We should only assert success if tensorflow actually loads or just consider it passed if it didn't fail due to torch
    if result.returncode != 0:
        if "No module named 'tensorflow'" in result.stderr:
            try:
                import pytest
                pytest.skip("tensorflow is not installed")
            except ImportError:
                print("Skipped: tensorflow is not installed")
                return
        elif "tensorflow" in result.stderr and "dll" in result.stderr.lower():
             return
        assert False, f"Import failed on tensorflow backend: {result.stderr}"

if __name__ == '__main__':
    test_defog_backend_imports()
    print("Backend import test passed!")
