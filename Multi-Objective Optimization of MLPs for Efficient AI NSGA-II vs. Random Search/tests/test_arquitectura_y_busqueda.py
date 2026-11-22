"""Pruebas de codificación, parámetros, presupuesto y equidad entre métodos."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.arquitectura import (Arquitectura, contar_parametros, cruce_uniforme, decodificar,
                              muestrear_genoma, muestrear_genoma_unico, mutar,
                              validar_arquitectura)
from src.busqueda_aleatoria import ejecutar_busqueda_aleatoria
from src.config import ConfigBusqueda, EspacioBusqueda
from src.datos import particionar
from src.evaluacion import (EvaluadorPresupuestado, PresupuestoAgotadoError,
                            evaluar_arquitectura)
from src.nsga2 import ejecutar_nsga2, validar_presupuesto_nsga2

ESP = EspacioBusqueda()


def test_formula_parametros_ejemplo_a_mano():
    # 64 -> 16 -> 10 : (64+1)*16 + (16+1)*10 = 1040 + 170 = 1210
    assert contar_parametros(Arquitectura((16,), "relu"), 64, 10) == 1210
    # 64 -> 8 -> 4 -> 10 : 65*8 + 9*4 + 5*10 = 520 + 36 + 50 = 606
    assert contar_parametros(Arquitectura((8, 4), "tanh"), 64, 10) == 606


def test_formula_coincide_con_sklearn_en_varias_arquitecturas():
    datos = particionar(0)
    for capas in [(2,), (7, 3), (20, 11, 5), (64, 64, 64)]:
        # evaluar_arquitectura lanza AssertionError si la fórmula no coincide
        r = evaluar_arquitectura(Arquitectura(capas, "relu"), datos)
        assert r.n_parametros == contar_parametros(Arquitectura(capas, "relu"), 64, 10)


def test_operadores_nunca_generan_invalidos():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        a, b = muestrear_genoma(rng, ESP), muestrear_genoma(rng, ESP)
        h = mutar(cruce_uniforme(a, b, rng), rng, ESP, sigma=1.5)
        validar_arquitectura(decodificar(h, ESP), ESP)   # no debe lanzar


def test_decodificar_rechaza_genomas_invalidos():
    with pytest.raises(ValueError):
        decodificar((4, 8, 8, 8, 0), ESP)       # 4 capas
    with pytest.raises(ValueError):
        decodificar((1, 1, 8, 8, 0), ESP)       # 1 unidad < mínimo
    with pytest.raises(ValueError):
        decodificar((1, 8, 8, 8, 5), ESP)       # activación inexistente
    with pytest.raises(ValueError):
        decodificar((1, 8, 8), ESP)             # longitud errónea


def test_muestreo_unico_no_repite():
    rng = np.random.default_rng(1)
    vistos = set()
    for _ in range(300):
        g = muestrear_genoma_unico(rng, ESP, vistos)
        arq = decodificar(g, ESP)
        assert arq not in vistos
        vistos.add(arq)


def test_validacion_de_presupuesto():
    assert validar_presupuesto_nsga2(200, 20) == 9
    assert validar_presupuesto_nsga2(60, 20) == 2
    with pytest.raises(ValueError):
        validar_presupuesto_nsga2(100, 30)    # (100-30) no es múltiplo de 30
    with pytest.raises(ValueError):
        validar_presupuesto_nsga2(10, 20)


def test_evaluador_limita_y_rechaza_duplicados():
    datos = particionar(0)
    ev = EvaluadorPresupuestado(datos, presupuesto=2)
    ev.evaluar(Arquitectura((4,), "relu"), 0)
    with pytest.raises(ValueError):
        ev.evaluar(Arquitectura((4,), "relu"), 0)         # duplicado
    ev.evaluar(Arquitectura((5,), "relu"), 0)
    with pytest.raises(PresupuestoAgotadoError):
        ev.evaluar(Arquitectura((6,), "relu"), 0)
    assert ev.n_evaluaciones == 2


def test_misma_cantidad_de_evaluaciones_y_poblacion_inicial_pareada():
    datos = particionar(3)
    cfg = ConfigBusqueda(presupuesto=40, tam_poblacion=10)
    e1 = EvaluadorPresupuestado(datos, 40)
    r1 = ejecutar_nsga2(e1, ESP, cfg, semilla=3)
    e2 = EvaluadorPresupuestado(datos, 40)
    r2 = ejecutar_busqueda_aleatoria(e2, ESP, semilla=3)
    assert len(r1.registro) == len(r2.registro) == 40
    assert len({f["arquitectura"] for f in r1.registro}) == 40      # sin duplicados
    assert len({f["arquitectura"] for f in r2.registro}) == 40
    # las primeras N coinciden en arquitectura Y en resultado (entrenamiento determinista)
    for a, b in zip(r1.registro[:10], r2.registro[:10]):
        assert a["arquitectura"] == b["arquitectura"]
        assert a["val_error"] == b["val_error"]
    assert all(f["generacion"] == 0 for f in r1.registro[:10])
    assert {f["generacion"] for f in r1.registro[10:]} == {1, 2, 3}


def test_reproducibilidad_misma_semilla_mismo_resultado():
    datos = particionar(5)
    cfg = ConfigBusqueda(presupuesto=30, tam_poblacion=10)
    runs = []
    for _ in range(2):
        ev = EvaluadorPresupuestado(datos, 30)
        runs.append(ejecutar_nsga2(ev, ESP, cfg, semilla=5).registro)
    for a, b in zip(*runs):
        assert a["arquitectura"] == b["arquitectura"] and a["val_error"] == b["val_error"]


def test_particiones_sin_solape_y_sin_fuga():
    datos = particionar(0)
    n = len(datos.y_ent) + len(datos.y_val) + len(datos.y_prueba)
    assert n == 1797
    assert abs(len(datos.y_ent) / n - 0.6) < 0.01
    # el escalador se ajustó con entrenamiento: media ~0 solo allí
    assert np.allclose(datos.X_ent.mean(axis=0), 0, atol=1e-8)
    assert not np.allclose(datos.X_val.mean(axis=0), 0, atol=1e-8)
