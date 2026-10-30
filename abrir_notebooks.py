"""Abre o Jupyter local; execute as células manualmente no navegador."""

import importlib.util
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    if importlib.util.find_spec("notebook") is None:
        sys.exit("Instale as dependências: python -m pip install -r requirements.txt")
    raiz = Path(__file__).resolve().parent
    print("Abrindo Jupyter no navegador. Para encerrar, use Ctrl+C neste terminal.", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "notebook", "--ip=127.0.0.1", "--ServerApp.open_browser=True", "notebooks/"],
            cwd=raiz, check=True,
        )
    except KeyboardInterrupt:
        pass
