"""Verificación de las piezas matemáticas de NSGA-II contra implementaciones de fuerza bruta."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pareto import (cobertura, distancia_crowding, domina, hipervolumen_2d,
                        mascara_no_dominados, ordenamiento_no_dominado,
                        seleccionar_supervivientes)


def test_domina_casos_basicos():
    assert domina([1, 1], [2, 2])
    assert domina([1, 2], [1, 3])          # igual en uno, mejor en otro
    assert not domina([1, 1], [1, 1])      # iguales: no hay dominancia
    assert not domina([1, 3], [2, 2])      # incomparables
    assert not domina([2, 2], [1, 3])


def test_orden_no_dominado_ejemplo_manual():
    F = np.array([[1, 5], [2, 3], [4, 1], [3, 4], [5, 5], [6, 6]], float)
    frentes = ordenamiento_no_dominado(F)
    assert sorted(frentes[0]) == [0, 1, 2]
    assert sorted(frentes[1]) == [3]
    assert sorted(frentes[2]) == [4]
    assert sorted(frentes[3]) == [5]


def _frentes_fuerza_bruta(F):
    restantes = set(range(len(F)))
    frentes = []
    while restantes:
        actual = [i for i in restantes
                  if not any(domina(F[j], F[i]) for j in restantes if j != i)]
        frentes.append(sorted(actual))
        restantes -= set(actual)
    return frentes


@pytest.mark.parametrize("semilla", range(20))
def test_orden_no_dominado_vs_fuerza_bruta(semilla):
    rng = np.random.default_rng(semilla)
    F = rng.integers(0, 8, size=(40, 2)).astype(float)   # muchos empates a propósito
    obtenido = [sorted(f) for f in ordenamiento_no_dominado(F)]
    assert obtenido == _frentes_fuerza_bruta(F)


def test_crowding_ejemplo_manual():
    F = np.array([[0, 4], [1, 2], [2, 1], [4, 0]], float)
    d = distancia_crowding(F)
    assert np.isinf(d[0]) and np.isinf(d[3])
    # interior 1: (2-0)/4 + (4-1)/4 = 0.5 + 0.75 ; interior 2: (4-1)/4 + (2-0)/4
    assert d[1] == pytest.approx(0.5 + 0.75)
    assert d[2] == pytest.approx(0.75 + 0.5)


def test_crowding_frente_pequeno_es_infinito():
    assert np.all(np.isinf(distancia_crowding(np.array([[1.0, 2.0], [2.0, 1.0]]))))


def test_hipervolumen_ejemplo_manual():
    # Un punto: área = (3-1)*(3-1) = 4
    assert hipervolumen_2d([[1, 1]], [3, 3]) == pytest.approx(4.0)
    # Dos puntos escalonados: 2x1 + 1x2 + solape descontado = 3
    assert hipervolumen_2d([[1, 2], [2, 1]], [3, 3]) == pytest.approx(3.0)
    # Punto dominado no cambia el valor
    assert hipervolumen_2d([[1, 2], [2, 1], [2.5, 2.5]], [3, 3]) == pytest.approx(3.0)
    # Punto fuera de la referencia no aporta
    assert hipervolumen_2d([[4, 1]], [3, 3]) == 0.0


@pytest.mark.parametrize("semilla", range(20))
def test_hipervolumen_vs_cuadricula(semilla):
    """Aproximación independiente: contar celdas dominadas de una malla fina."""
    rng = np.random.default_rng(semilla)
    F = rng.uniform(0, 1, size=(12, 2))
    ref = np.array([1.1, 1.1])
    n = 800
    xs = (np.arange(n) + 0.5) / n * ref[0]
    ys = (np.arange(n) + 0.5) / n * ref[1]
    gx, gy = np.meshgrid(xs, ys)
    dominado = np.zeros_like(gx, dtype=bool)
    for f1, f2 in F:
        dominado |= (gx >= f1) & (gy >= f2)
    aprox = dominado.mean() * ref[0] * ref[1]
    assert hipervolumen_2d(F, ref) == pytest.approx(aprox, abs=5e-3)


def test_hipervolumen_invariante_a_duplicados_y_orden():
    F = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    base = hipervolumen_2d(F, [1, 1])
    assert hipervolumen_2d(np.vstack([F, F[1]]), [1, 1]) == pytest.approx(base)
    assert hipervolumen_2d(F[::-1], [1, 1]) == pytest.approx(base)


def test_cobertura():
    A = np.array([[1, 1]])
    B = np.array([[2, 2], [0, 5]])
    assert cobertura(A, B) == 0.5
    assert cobertura(B, A) == 0.0


def test_supervivientes_elitismo_y_tamano():
    rng = np.random.default_rng(3)
    F = rng.uniform(size=(30, 2))
    idx, rango, crowd = seleccionar_supervivientes(F, 10)
    assert len(idx) == 10 and len(set(idx)) == 10
    frentes = ordenamiento_no_dominado(F)
    # Si el primer frente cabe, TODOS sus miembros sobreviven (elitismo)
    if len(frentes[0]) <= 10:
        assert set(frentes[0]) <= set(idx)
    # Ningún elegido pertenece a un frente peor que algún no elegido de frente mejor
    peor_elegido = max(rango[idx])
    no_elegidos = [i for i in range(30) if i not in set(idx)]
    assert all(rango[i] >= peor_elegido for i in no_elegidos)


def test_mascara_no_dominados_coincide_con_frente0():
    rng = np.random.default_rng(0)
    F = rng.integers(0, 6, size=(25, 2)).astype(float)
    m = mascara_no_dominados(F)
    assert sorted(np.where(m)[0]) == sorted(ordenamiento_no_dominado(F)[0])


def test_rechaza_no_finitos():
    with pytest.raises(ValueError):
        ordenamiento_no_dominado([[1.0, np.nan], [2.0, 1.0]])


def test_wilcoxon_exacto_coincide_con_scipy_sin_empates():
    from scipy import stats
    from src.metricas import wilcoxon_pareado_exacto
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(5, 13))
        x, y = rng.normal(size=n), rng.normal(size=n) + rng.normal(0, 0.5)
        esperado = stats.wilcoxon(x, y, method="exact").pvalue
        assert wilcoxon_pareado_exacto(x, y) == pytest.approx(esperado, abs=1e-12)


def test_wilcoxon_exacto_casos_conocidos():
    from src.metricas import wilcoxon_pareado_exacto
    assert wilcoxon_pareado_exacto([1, 2, 3], [1, 2, 3]) == 1.0                    # todo empate
    # n = 10, todas las diferencias del mismo signo: p = 2 / 2^10
    assert wilcoxon_pareado_exacto(np.arange(1, 11) + 0.5, np.zeros(10)) == pytest.approx(2 / 1024)
    # los empates (diferencias nulas) se descartan: 4 pares con el mismo signo -> p = 2 / 2^4
    assert wilcoxon_pareado_exacto([1, 2, 3, 4, 5, 5], [0, 0, 0, 0, 5, 5]) == pytest.approx(2 / 16)
