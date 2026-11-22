"""Configuración global del experimento (única fuente de verdad).

Todos los valores que definen el protocolo experimental están aquí, de modo que
cambiar un número no exige tocar la lógica del algoritmo.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EspacioBusqueda:
    """Límites del espacio de arquitecturas MLP (impide redes excesivamente grandes)."""
    capas_min: int = 1
    capas_max: int = 3
    unidades_min: int = 2
    unidades_max: int = 64
    activaciones: tuple = ("relu", "tanh")


@dataclass(frozen=True)
class ConfigEntrenamiento:
    """Procedimiento de entrenamiento idéntico para TODAS las arquitecturas y métodos."""
    max_iter: int = 100            # épocas máximas (presupuesto fijo de entrenamiento)
    tasa_aprendizaje: float = 0.01
    solver: str = "adam"


@dataclass(frozen=True)
class ConfigBusqueda:
    """Parámetros de NSGA-II y presupuesto compartido con la búsqueda aleatoria."""
    presupuesto: int = 200         # nº de arquitecturas entrenadas (evaluaciones reales)
    tam_poblacion: int = 20        # N de NSGA-II
    prob_cruce: float = 0.9
    sigma_mutacion: float = 0.6    # desviación (escala log) de la mutación de unidades
    max_intentos_hijo: int = 100   # reintentos para obtener un hijo no duplicado


@dataclass(frozen=True)
class ConfigDatos:
    """Datos: Digits de scikit-learn, partición estratificada 60/20/20."""
    frac_validacion: float = 0.20
    frac_prueba: float = 0.20


@dataclass(frozen=True)
class ConfigMetricas:
    """Punto de referencia COMÚN para el hipervolumen (se fija a priori)."""
    error_ref: float = 0.90        # error esperado al adivinar en 10 clases balanceadas
    tolerancia_compacta: float = 0.01  # 1 punto porcentual para la solución "compacta"


SEMILLAS_POR_DEFECTO = tuple(range(10))
