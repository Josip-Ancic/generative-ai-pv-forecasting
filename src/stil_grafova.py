from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAPA = Path(__file__).resolve().parent.parent
MAPA_PODACI = MAPA / "podaci"
MAPA_SLIKE = MAPA / "slike"

ZELENA = "#2e9e5b"
TAMNOZELENA = "#1a6b3c"
NARANCASTA = "#e2622a"
SIVA = "#888888"


def postavi_stil():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3})
    MAPA_SLIKE.mkdir(exist_ok=True)


def spremi(figura, naziv):
    figura.tight_layout()
    figura.savefig(MAPA_SLIKE / naziv)
    plt.close(figura)
