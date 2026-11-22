#!/usr/bin/env python3
"""Genera Notebook_Investigacion_NSGA2.ipynb a partir de las partes en docs/ (no ejecuta nada)."""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent / "docs"))
from _nb_parte1 import parte1
from _nb_parte2 import parte2

try:
    from _nb_parte3 import INTERPRETACIONES, parte3
except ImportError:           # primera pasada: sin interpretaciones todavía
    INTERPRETACIONES = {k: "*(interpretación pendiente)*" for k in
                        ["resumen", "fig123", "fig4", "fig6", "stats", "test", "sens"]}
    parte3 = lambda md, code: []


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


def main(salida="Notebook_Investigacion_NSGA2.ipynb"):
    nb = nbf.v4.new_notebook()
    nb.cells = parte1(md, code) + parte2(md, code, INTERPRETACIONES) + parte3(md, code)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, salida)
    print(f"{salida}: {len(nb.cells)} celdas")


if __name__ == "__main__":
    main(*sys.argv[1:])
