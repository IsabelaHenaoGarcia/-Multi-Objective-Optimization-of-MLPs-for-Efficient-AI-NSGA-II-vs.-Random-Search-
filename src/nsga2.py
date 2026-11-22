"""NSGA-II (Deb et al., 2002) implementado desde cero para arquitecturas MLP.

Contabilidad del presupuesto (distinguimos iteraciones de evaluaciones):
    evaluaciones = N (población inicial) + G * N (G generaciones, N hijos cada una)
    => presupuesto B = N * (1 + G); se exige que (B - N) sea múltiplo de N.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .arquitectura import (cruce_uniforme, decodificar, muestrear_genoma_unico, mutar)
from .config import ConfigBusqueda, EspacioBusqueda
from .pareto import distancia_crowding, ordenamiento_no_dominado, seleccionar_supervivientes


@dataclass
class ResultadoBusqueda:
    metodo: str
    semilla: int
    registro: list            # una fila por arquitectura evaluada (en orden)
    tiempo_s: float
    info: dict = field(default_factory=dict)


def validar_presupuesto_nsga2(presupuesto, tam_poblacion):
    """Devuelve G (nº de generaciones) o lanza ValueError si el presupuesto no encaja."""
    if tam_poblacion < 2:
        raise ValueError("La población debe tener al menos 2 individuos")
    if presupuesto < tam_poblacion:
        raise ValueError("El presupuesto es menor que la población inicial")
    if (presupuesto - tam_poblacion) % tam_poblacion != 0:
        raise ValueError(
            f"Presupuesto {presupuesto} incompatible con N={tam_poblacion}: "
            "(B - N) debe ser múltiplo de N para igualar evaluaciones exactamente")
    return (presupuesto - tam_poblacion) // tam_poblacion


def _rango_y_crowding(F):
    """Rango (0 = mejor frente) y crowding de cada individuo de la población."""
    rango = np.zeros(len(F), dtype=int)
    crowd = np.zeros(len(F))
    for r, frente in enumerate(ordenamiento_no_dominado(F)):
        rango[frente] = r
        crowd[frente] = distancia_crowding(F[frente])
    return rango, crowd


def _torneo_binario(rango, crowd, rng):
    """Comparación por crowding: menor rango gana; si empatan, mayor crowding."""
    i, j = rng.choice(len(rango), size=2, replace=False)
    if rango[i] != rango[j]:
        return int(i if rango[i] < rango[j] else j)
    if crowd[i] != crowd[j]:
        return int(i if crowd[i] > crowd[j] else j)
    return int(rng.choice([i, j]))


def _generar_hijo(genomas, rango, crowd, rng, espacio, cfg, vistos):
    """Torneo -> cruce -> mutación, reintentando hasta un fenotipo NUEVO.

    Devuelve (genoma, fue_fallback). Si tras max_intentos_hijo no se logra, se usa una
    muestra aleatoria única (se contabiliza para reportarlo, no se oculta).
    """
    for _ in range(cfg.max_intentos_hijo):
        p1 = genomas[_torneo_binario(rango, crowd, rng)]
        p2 = genomas[_torneo_binario(rango, crowd, rng)]
        hijo = cruce_uniforme(p1, p2, rng) if rng.random() < cfg.prob_cruce else p1
        hijo = mutar(hijo, rng, espacio, cfg.sigma_mutacion)
        if decodificar(hijo, espacio) not in vistos:
            return hijo, False
    return muestrear_genoma_unico(rng, espacio, vistos), True


def _objetivos(registro, desde):
    """Matriz (error, parámetros) de las evaluaciones registradas desde `desde`."""
    return np.array([[f["val_error"], f["n_parametros"]] for f in registro[desde:]], dtype=float)


def ejecutar_nsga2(evaluador, espacio=EspacioBusqueda(), cfg=ConfigBusqueda(), semilla=0):
    """Ejecuta NSGA-II consumiendo exactamente `evaluador.presupuesto` evaluaciones."""
    N = cfg.tam_poblacion
    G = validar_presupuesto_nsga2(evaluador.presupuesto, N)
    if evaluador.n_evaluaciones != 0:
        raise ValueError("El evaluador debe estar vacío al iniciar")
    t0 = time.perf_counter()
    rng_init = np.random.default_rng([semilla, 0])   # mismo flujo que la búsqueda aleatoria
    rng = np.random.default_rng([semilla, 1])        # operadores evolutivos

    # 1) Población inicial: las primeras N muestras aleatorias únicas.
    genomas = []
    for _ in range(N):
        g = muestrear_genoma_unico(rng_init, espacio, evaluador.vistos)
        evaluador.evaluar(decodificar(g, espacio), generacion=0)
        genomas.append(g)
    F = _objetivos(evaluador.registro, 0)

    # 2) Generaciones: N hijos evaluados por generación + reemplazo elitista (2N -> N).
    fallbacks = 0
    for gen in range(1, G + 1):
        rango, crowd = _rango_y_crowding(F)
        hijos = []
        inicio = evaluador.n_evaluaciones
        for _ in range(N):
            hijo, fb = _generar_hijo(genomas, rango, crowd, rng, espacio, cfg, evaluador.vistos)
            fallbacks += int(fb)
            evaluador.evaluar(decodificar(hijo, espacio), generacion=gen)
            hijos.append(hijo)
        F_hijos = _objetivos(evaluador.registro, inicio)
        F_comb = np.vstack([F, F_hijos])
        elegidos, _, _ = seleccionar_supervivientes(F_comb, N)
        genomas_comb = genomas + hijos
        genomas = [genomas_comb[i] for i in elegidos]
        F = F_comb[elegidos]

    if evaluador.n_evaluaciones != evaluador.presupuesto:
        raise AssertionError("NSGA-II no consumió exactamente el presupuesto")
    return ResultadoBusqueda("NSGA-II", semilla, evaluador.registro, time.perf_counter() - t0,
                             {"generaciones": G, "hijos_fallback_aleatorio": fallbacks})
