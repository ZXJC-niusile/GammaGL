"""CPU-isolated Frechet ChemNet Distance helpers for DeFoG evaluation."""

import os
import pickle
import subprocess
import sys
import tempfile


def _compute_fcd_cpu(generated_smiles, reference_smiles):
    """Compute FCD after hiding CUDA from the current process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from fcd import get_fcd
    except ImportError as exc:
        raise RuntimeError(
            "fcd package not installed. Install with: pip install fcd"
        ) from exc

    return float(get_fcd(generated_smiles, reference_smiles, device="cpu"))


def _run_worker(input_path, output_path):
    with open(input_path, "rb") as file:
        payload = pickle.load(file)

    score = _compute_fcd_cpu(
        payload["generated_smiles"],
        payload["reference_smiles"],
    )
    with open(output_path, "wb") as file:
        pickle.dump(score, file)


def compute_fcd_cpu_isolated(generated_smiles, reference_smiles):
    """Compute FCD in a CPU-only subprocess.

    The ``fcd`` package may create DataLoader workers internally. Running it in
    a subprocess where CUDA is hidden before import avoids fork-after-CUDA
    failures when evaluation is launched from a GPU training process.
    """
    payload = {
        "generated_smiles": [s for s in generated_smiles if s is not None],
        "reference_smiles": reference_smiles,
    }

    with tempfile.TemporaryDirectory(prefix="defog_fcd_") as tmpdir:
        input_path = os.path.join(tmpdir, "input.pkl")
        output_path = os.path.join(tmpdir, "output.pkl")
        with open(input_path, "wb") as file:
            pickle.dump(payload, file)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [sys.executable, __file__, "--worker", input_path, output_path],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(message)

        with open(output_path, "rb") as file:
            return pickle.load(file)


def _main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2], sys.argv[3])
        return
    raise SystemExit("Unsupported defog_fcd.py invocation")


if __name__ == "__main__":
    _main()
