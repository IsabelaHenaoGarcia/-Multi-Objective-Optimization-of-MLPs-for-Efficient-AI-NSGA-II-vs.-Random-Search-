#!/usr/bin/env python3
"""Ejecuta el protocolo experimental completo y guarda resultados en `resultados/`.

Uso:
    python ejecutar_experimentos.py --modo completo     # protocolo del informe (~30 min en 1 CPU)
    python ejecutar_experimentos.py --modo rapido       # demostración (2 semillas, presupuesto 60)
Es reanudable: las corridas ya guardadas se omiten (use --rehacer para repetirlas).
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from src.config import ConfigBusqueda, ConfigEntrenamiento, EspacioBusqueda, SEMILLAS_POR_DEFECTO
from src.experimento import ejecutar_experimento, evaluar_referencias_manuales

PROTOCOLOS = {
    # tag: (configuración de búsqueda, métodos)
    "completo": {
        "principal": (ConfigBusqueda(presupuesto=200, tam_poblacion=20), ("NSGA-II", "Aleatoria")),
        "sens_pob10": (ConfigBusqueda(presupuesto=200, tam_poblacion=10), ("NSGA-II",)),
        "sens_pob50": (ConfigBusqueda(presupuesto=200, tam_poblacion=50), ("NSGA-II",)),
    },
    "rapido": {
        "rapido": (ConfigBusqueda(presupuesto=60, tam_poblacion=20), ("NSGA-II", "Aleatoria")),
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=PROTOCOLOS, default="completo")
    ap.add_argument("--salida", default=None, help="carpeta de resultados (por defecto resultados/ o resultados_rapido/)")
    ap.add_argument("--semillas", type=int, default=None, help="nº de semillas (por defecto 10 / 2)")
    ap.add_argument("--rehacer", action="store_true")
    a = ap.parse_args()
    salida = Path(a.salida or ("resultados" if a.modo == "completo" else "resultados_rapido"))
    n_sem = a.semillas or (len(SEMILLAS_POR_DEFECTO) if a.modo == "completo" else 2)
    semillas = list(range(n_sem))
    salida.mkdir(parents=True, exist_ok=True)

    (salida / "entorno.json").write_text(json.dumps({
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "sistema": platform.platform(),
        "semillas": semillas, "modo": a.modo,
        "espacio": EspacioBusqueda().__dict__, "entrenamiento": ConfigEntrenamiento().__dict__,
        "protocolos": {t: {"config": c.__dict__, "metodos": list(m)} for t, (c, m) in PROTOCOLOS[a.modo].items()},
    }, indent=2, default=str))

    for tag, (cfg, metodos) in PROTOCOLOS[a.modo].items():
        ejecutar_experimento(tag, cfg, metodos, semillas, salida, reanudar=not a.rehacer)
    ref = evaluar_referencias_manuales(semillas)
    ref.to_csv(salida / "referencias_manuales.csv", index=False)
    print("Listo. Resultados en", salida.resolve())


if __name__ == "__main__":
    main()
