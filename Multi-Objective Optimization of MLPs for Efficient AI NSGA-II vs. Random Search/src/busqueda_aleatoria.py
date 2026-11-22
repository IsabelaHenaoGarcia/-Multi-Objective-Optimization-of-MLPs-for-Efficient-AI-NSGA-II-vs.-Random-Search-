"""Búsqueda aleatoria: método de referencia con el MISMO espacio, muestreador y presupuesto."""
import time

import numpy as np

from .arquitectura import decodificar, muestrear_genoma_unico
from .config import EspacioBusqueda
from .nsga2 import ResultadoBusqueda


def ejecutar_busqueda_aleatoria(evaluador, espacio=EspacioBusqueda(), semilla=0):
    """Evalúa `presupuesto` arquitecturas distintas muestreadas al azar.

    Usa el mismo generador (semilla, 0) que la población inicial de NSGA-II, de modo
    que sus primeras N evaluaciones coinciden (diseño pareado): las diferencias
    posteriores se deben únicamente a cómo se invierte el resto del presupuesto.
    """
    if evaluador.n_evaluaciones != 0:
        raise ValueError("El evaluador debe estar vacío al iniciar")
    t0 = time.perf_counter()
    rng_init = np.random.default_rng([semilla, 0])
    for _ in range(evaluador.presupuesto):
        g = muestrear_genoma_unico(rng_init, espacio, evaluador.vistos)
        evaluador.evaluar(decodificar(g, espacio), generacion=0)
    if evaluador.n_evaluaciones != evaluador.presupuesto:
        raise AssertionError("La búsqueda aleatoria no consumió exactamente el presupuesto")
    return ResultadoBusqueda("Aleatoria", semilla, evaluador.registro, time.perf_counter() - t0, {})
