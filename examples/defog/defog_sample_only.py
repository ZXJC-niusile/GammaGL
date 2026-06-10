"""Sample and evaluate a saved DeFoG checkpoint."""

import os
import subprocess
import sys


def main():
    trainer = os.path.join(os.path.dirname(__file__), 'defog_trainer.py')
    cmd = [sys.executable, trainer, '--sample_only', *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, env=os.environ.copy()))


if __name__ == '__main__':
    main()
