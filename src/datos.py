"""Carga y partición de datos (Digits de scikit-learn) sin fuga de información."""
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ConfigDatos


@dataclass(frozen=True)
class DatosParticionados:
    """Tres particiones estratificadas. `prueba` solo se usa en la evaluación final."""
    X_ent: np.ndarray
    y_ent: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_prueba: np.ndarray
    y_prueba: np.ndarray
    n_entrada: int
    n_clases: int
    semilla: int


def cargar_digits():
    """Digits (UCI Optical Recognition of Handwritten Digits, test set): 1797 x 64."""
    X, y = load_digits(return_X_y=True)
    if not np.all(np.isfinite(X)):
        raise ValueError("El conjunto de datos contiene valores faltantes o no finitos")
    if len(X) != len(y):
        raise ValueError("X e y tienen distinto número de filas")
    return X.astype(float), y.astype(int)


def particionar(semilla, cfg=ConfigDatos()):
    """Partición estratificada entrenamiento/validación/prueba (60/20/20).

    El escalador se AJUSTA SOLO con entrenamiento y luego se aplica a validación y
    prueba (evita fuga de datos). La semilla fija la partición.
    """
    X, y = cargar_digits()
    X_ent, X_resto, y_ent, y_resto = train_test_split(
        X, y, test_size=cfg.frac_validacion + cfg.frac_prueba, stratify=y, random_state=semilla)
    prop_prueba = cfg.frac_prueba / (cfg.frac_validacion + cfg.frac_prueba)
    X_val, X_prueba, y_val, y_prueba = train_test_split(
        X_resto, y_resto, test_size=prop_prueba, stratify=y_resto, random_state=semilla)
    escalador = StandardScaler().fit(X_ent)
    X_ent, X_val, X_prueba = (escalador.transform(a) for a in (X_ent, X_val, X_prueba))
    n_clases = int(len(np.unique(y)))
    if X_ent.shape[1] != X_val.shape[1] or X_ent.shape[1] != X_prueba.shape[1]:
        raise ValueError("Dimensiones inconsistentes entre particiones")
    return DatosParticionados(X_ent, y_ent, X_val, y_val, X_prueba, y_prueba,
                              X_ent.shape[1], n_clases, semilla)
