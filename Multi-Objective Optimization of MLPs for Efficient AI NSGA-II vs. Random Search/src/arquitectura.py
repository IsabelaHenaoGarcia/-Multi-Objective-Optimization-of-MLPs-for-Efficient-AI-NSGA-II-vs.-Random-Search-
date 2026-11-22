"""Codificación de arquitecturas MLP, operadores genéticos y conteo de parámetros.

Genoma (tupla de enteros): (n_capas, u_1, ..., u_capas_max, indice_activacion).
Solo las primeras n_capas unidades están "activas" (fenotipo); las demás son genes
latentes. Decodificar SIEMPRE produce una arquitectura válida, y los operadores
(cruce, mutación) recortan al rango permitido, por lo que no generan inválidas.
"""
from dataclasses import dataclass

import numpy as np

from .config import EspacioBusqueda


@dataclass(frozen=True)
class Arquitectura:
    """Fenotipo: lo que realmente se entrena. Hashable (sirve para detectar duplicados)."""
    capas: tuple          # unidades por capa oculta, p.ej. (16, 8)
    activacion: str       # "relu" | "tanh"

    def __str__(self):
        return f"{list(self.capas)}-{self.activacion}"


def validar_arquitectura(arq, espacio):
    """Lanza ValueError si la arquitectura sale del espacio de búsqueda."""
    if not (espacio.capas_min <= len(arq.capas) <= espacio.capas_max):
        raise ValueError(f"Número de capas inválido: {len(arq.capas)}")
    for u in arq.capas:
        if not (espacio.unidades_min <= u <= espacio.unidades_max):
            raise ValueError(f"Unidades fuera de rango: {u}")
    if arq.activacion not in espacio.activaciones:
        raise ValueError(f"Activación no permitida: {arq.activacion}")


def contar_parametros(arq, n_entrada, n_salida):
    """Parámetros entrenables de una MLP densa (pesos + sesgos).

    P = sum_{l=1}^{L+1} (n_{l-1} + 1) * n_l, con n_0 = n_entrada, n_{L+1} = n_salida.
    """
    if n_entrada < 1 or n_salida < 1:
        raise ValueError("n_entrada y n_salida deben ser >= 1")
    tamanos = [n_entrada, *arq.capas, n_salida]
    return int(sum((a + 1) * b for a, b in zip(tamanos[:-1], tamanos[1:])))


def longitud_genoma(espacio):
    return 1 + espacio.capas_max + 1


def decodificar(genoma, espacio):
    """Genoma -> Arquitectura (validando rangos; no corrige silenciosamente)."""
    if len(genoma) != longitud_genoma(espacio):
        raise ValueError(f"Longitud de genoma incorrecta: {len(genoma)}")
    n_capas = genoma[0]
    unidades = genoma[1:1 + espacio.capas_max]
    idx_act = genoma[-1]
    if not (espacio.capas_min <= n_capas <= espacio.capas_max):
        raise ValueError(f"n_capas fuera de rango: {n_capas}")
    if not (0 <= idx_act < len(espacio.activaciones)):
        raise ValueError(f"Índice de activación inválido: {idx_act}")
    arq = Arquitectura(tuple(int(u) for u in unidades[:n_capas]), espacio.activaciones[idx_act])
    validar_arquitectura(arq, espacio)
    return arq


def _unidades_aleatorias(rng, espacio):
    """Unidades con distribución log-uniforme: cubre redes pequeñas y grandes."""
    log_u = rng.uniform(np.log(espacio.unidades_min), np.log(espacio.unidades_max))
    return int(np.clip(round(float(np.exp(log_u))), espacio.unidades_min, espacio.unidades_max))


def muestrear_genoma(rng, espacio):
    """Genoma aleatorio válido (muestreador común a NSGA-II inicial y búsqueda aleatoria)."""
    n_capas = int(rng.integers(espacio.capas_min, espacio.capas_max + 1))
    unidades = [_unidades_aleatorias(rng, espacio) for _ in range(espacio.capas_max)]
    idx_act = int(rng.integers(len(espacio.activaciones)))
    return (n_capas, *unidades, idx_act)


def muestrear_genoma_unico(rng, espacio, vistos, max_intentos=10_000):
    """Muestrea hasta obtener un fenotipo no evaluado antes (`vistos`: set de Arquitectura)."""
    for _ in range(max_intentos):
        g = muestrear_genoma(rng, espacio)
        if decodificar(g, espacio) not in vistos:
            return g
    raise RuntimeError("No se encontró una arquitectura nueva: espacio casi agotado")


def cruce_uniforme(g1, g2, rng):
    """Cada gen se hereda de un padre al azar (p=0.5). Cualquier mezcla es válida."""
    mascara = rng.random(len(g1)) < 0.5
    hijo = tuple(int(a if m else b) for a, b, m in zip(g1, g2, mascara))
    return hijo


def mutar(genoma, rng, espacio, sigma=0.6):
    """Mutación por gen con probabilidad 1/longitud (≈1 gen cambia en promedio).

    - n_capas: ±1.
    - unidades: se multiplican por exp(N(0, sigma)) (cambio relativo, escala log).
    - activación: se cambia por otra permitida.
    Todo se recorta a los límites, así que el resultado es siempre válido.
    """
    p = 1.0 / len(genoma)
    g = list(genoma)
    if rng.random() < p:
        g[0] = int(np.clip(g[0] + rng.choice([-1, 1]), espacio.capas_min, espacio.capas_max))
    for i in range(1, 1 + espacio.capas_max):
        if rng.random() < p:
            nuevo = round(g[i] * float(np.exp(rng.normal(0.0, sigma))))
            g[i] = int(np.clip(nuevo, espacio.unidades_min, espacio.unidades_max))
    if rng.random() < p:
        otras = [k for k in range(len(espacio.activaciones)) if k != g[-1]]
        g[-1] = int(rng.choice(otras))
    return tuple(g)
