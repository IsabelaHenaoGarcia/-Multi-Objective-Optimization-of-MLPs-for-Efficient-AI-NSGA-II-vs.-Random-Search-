"""Entrenamiento/evaluación de una arquitectura y evaluador con presupuesto estricto."""
import time
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from .arquitectura import contar_parametros, validar_arquitectura
from .config import ConfigEntrenamiento


class PresupuestoAgotadoError(RuntimeError):
    """Se intentó evaluar más arquitecturas que el presupuesto permitido."""


@dataclass(frozen=True)
class ResultadoEvaluacion:
    val_error: float        # objetivo 1: 1 - accuracy en validación
    val_accuracy: float
    n_parametros: int       # objetivo 2
    convergio: bool         # False si se agotó max_iter (ConvergenceWarning)
    tiempo_s: float


def _construir_modelo(arq, cfg, semilla):
    return MLPClassifier(hidden_layer_sizes=arq.capas, activation=arq.activacion,
                         solver=cfg.solver, learning_rate_init=cfg.tasa_aprendizaje,
                         max_iter=cfg.max_iter, random_state=semilla)


def _entrenar(arq, datos, cfg, semilla):
    """Entrena y devuelve (modelo, convergió, tiempo). Mismo procedimiento siempre."""
    modelo = _construir_modelo(arq, cfg, semilla)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo.fit(datos.X_ent, datos.y_ent)
    convergio = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
    return modelo, convergio, time.perf_counter() - t0


def _parametros_reales(modelo):
    return int(sum(c.size for c in modelo.coefs_) + sum(b.size for b in modelo.intercepts_))


def n_salida_mlp(n_clases):
    """Neuronas de salida que usa scikit-learn: 1 si binario, n_clases si multiclase."""
    return 1 if n_clases == 2 else n_clases


def evaluar_arquitectura(arq, datos, cfg=ConfigEntrenamiento()):
    """Entrena con ENTRENAMIENTO y mide en VALIDACIÓN. No toca el conjunto de prueba."""
    modelo, convergio, t = _entrenar(arq, datos, cfg, datos.semilla)
    n_param = contar_parametros(arq, datos.n_entrada, n_salida_mlp(datos.n_clases))
    if n_param != _parametros_reales(modelo):
        raise AssertionError("La fórmula de parámetros no coincide con el modelo de sklearn")
    acc = float(modelo.score(datos.X_val, datos.y_val))
    if not np.isfinite(acc):
        raise ValueError(f"Resultado no finito para {arq}")
    return ResultadoEvaluacion(1.0 - acc, acc, n_param, convergio, t)


def evaluar_en_prueba(arq, datos, cfg=ConfigEntrenamiento()):
    """EVALUACIÓN FINAL (única función que usa prueba). Reentrena con la misma semilla."""
    modelo, _, _ = _entrenar(arq, datos, cfg, datos.semilla)
    val_error = 1.0 - float(modelo.score(datos.X_val, datos.y_val))
    return 1.0 - float(modelo.score(datos.X_prueba, datos.y_prueba)), val_error


class EvaluadorPresupuestado:
    """Cuenta evaluaciones REALES (un entrenamiento = una evaluación) y las limita.

    Ambos métodos usan esta misma clase: el presupuesto se iguala por construcción.
    Rechaza duplicados (el resultado es determinista, repetirlos desperdicia presupuesto).
    """

    def __init__(self, datos, presupuesto, cfg=ConfigEntrenamiento()):
        if presupuesto < 1:
            raise ValueError("El presupuesto debe ser >= 1")
        self.datos, self.presupuesto, self.cfg = datos, presupuesto, cfg
        self.registro = []
        self._vistos = set()

    @property
    def n_evaluaciones(self):
        return len(self.registro)

    @property
    def vistos(self):
        return self._vistos

    def evaluar(self, arq, generacion):
        if self.n_evaluaciones >= self.presupuesto:
            raise PresupuestoAgotadoError(f"Presupuesto de {self.presupuesto} agotado")
        if arq in self._vistos:
            raise ValueError(f"Arquitectura duplicada: {arq}")
        r = evaluar_arquitectura(arq, self.datos, self.cfg)
        self._vistos.add(arq)
        self.registro.append({
            "orden": self.n_evaluaciones + 1, "generacion": generacion, "arquitectura": arq,
            "val_error": r.val_error, "val_accuracy": r.val_accuracy,
            "n_parametros": r.n_parametros, "convergio": r.convergio, "tiempo_eval_s": r.tiempo_s})
        return r
