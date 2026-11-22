"""Ejecución de corridas, registro en disco (CSV) y evaluación final en prueba."""
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .arquitectura import Arquitectura
from .busqueda_aleatoria import ejecutar_busqueda_aleatoria
from .config import ConfigBusqueda, ConfigEntrenamiento, EspacioBusqueda
from .datos import particionar
from .evaluacion import EvaluadorPresupuestado, evaluar_en_prueba
from .nsga2 import ejecutar_nsga2

METODOS = ("NSGA-II", "Aleatoria")

# Referencias manuales (NO forman parte del presupuesto de búsqueda; solo contextualizan).
REFERENCIAS_MANUALES = {
    "manual_pequena": Arquitectura((16,), "relu"),
    "manual_mediana": Arquitectura((64,), "relu"),
    "manual_grande": Arquitectura((64, 64, 64), "relu"),   # la mayor del espacio
}


def a_texto(arq):
    return "-".join(str(u) for u in arq.capas), arq.activacion


def desde_texto(capas, activacion):
    return Arquitectura(tuple(int(u) for u in str(capas).split("-")), str(activacion))


def registro_a_dataframe(registro, tag, metodo, semilla):
    filas = []
    for f in registro:
        capas, act = a_texto(f["arquitectura"])
        filas.append({"tag": tag, "metodo": metodo, "semilla": semilla, "orden": f["orden"],
                      "generacion": f["generacion"], "capas": capas, "n_capas": len(f["arquitectura"].capas),
                      "activacion": act, "val_error": f["val_error"], "val_accuracy": f["val_accuracy"],
                      "n_parametros": f["n_parametros"], "convergio": f["convergio"],
                      "tiempo_eval_s": f["tiempo_eval_s"]})
    return pd.DataFrame(filas)


def ejecutar_corrida(metodo, semilla, cfg_busqueda, cfg_ent=ConfigEntrenamiento(),
                     espacio=EspacioBusqueda()):
    """Una corrida de un método con una semilla. Devuelve (DataFrame de evaluaciones, resultado)."""
    if metodo not in METODOS:
        raise ValueError(f"Método desconocido: {metodo}")
    datos = particionar(semilla)
    evaluador = EvaluadorPresupuestado(datos, cfg_busqueda.presupuesto, cfg_ent)
    if metodo == "NSGA-II":
        res = ejecutar_nsga2(evaluador, espacio, cfg_busqueda, semilla)
    else:
        res = ejecutar_busqueda_aleatoria(evaluador, espacio, semilla)
    if evaluador.n_evaluaciones != cfg_busqueda.presupuesto:
        raise AssertionError("Número real de evaluaciones distinto del presupuesto")
    return registro_a_dataframe(res.registro, "", metodo, semilla), res


def ruta_corrida(dir_salida, tag, metodo, semilla):
    return Path(dir_salida) / "corridas" / f"{tag}__{metodo}__s{semilla}"


def ejecutar_experimento(tag, cfg_busqueda, metodos, semillas, dir_salida, cfg_ent=ConfigEntrenamiento(),
                         espacio=EspacioBusqueda(), reanudar=True, verbose=True):
    """Ejecuta todas las (método, semilla) y guarda cada corrida; reanuda si ya existen."""
    for semilla in semillas:
        for metodo in metodos:
            base = ruta_corrida(dir_salida, tag, metodo, semilla)
            base.parent.mkdir(parents=True, exist_ok=True)
            csv, meta = base.with_suffix(".csv"), base.with_suffix(".json")
            if reanudar and csv.exists() and meta.exists():
                continue
            df, res = ejecutar_corrida(metodo, semilla, cfg_busqueda, cfg_ent, espacio)
            df["tag"] = tag
            df.to_csv(csv, index=False)
            meta.write_text(json.dumps({"tag": tag, "metodo": metodo, "semilla": semilla,
                                        "tiempo_s": res.tiempo_s, "info": res.info,
                                        "evaluaciones": len(df), "config_busqueda": asdict(cfg_busqueda),
                                        "config_entrenamiento": asdict(cfg_ent)}, indent=2))
            if verbose:
                print(f"[{tag}] {metodo:9s} semilla {semilla}: {len(df)} evaluaciones, {res.tiempo_s:.1f} s", flush=True)


def cargar_resultados(dir_salida, tags=None):
    """Lee todas las corridas guardadas -> (evaluaciones, ejecuciones) como DataFrames."""
    base = Path(dir_salida) / "corridas"
    evals, metas = [], []
    for meta in sorted(base.glob("*.json")):
        m = json.loads(meta.read_text())
        if tags is not None and m["tag"] not in tags:
            continue
        evals.append(pd.read_csv(meta.with_suffix(".csv")))
        metas.append({"tag": m["tag"], "metodo": m["metodo"], "semilla": m["semilla"],
                      "tiempo_s": m["tiempo_s"], "evaluaciones": m["evaluaciones"],
                      "fallback_aleatorio": m["info"].get("hijos_fallback_aleatorio", 0)})
    if not evals:
        raise FileNotFoundError(f"No hay corridas en {base}")
    return pd.concat(evals, ignore_index=True), pd.DataFrame(metas)


def evaluar_referencias_manuales(semillas, cfg_ent=ConfigEntrenamiento()):
    """Entrena las referencias manuales con cada partición y las mide en validación y prueba."""
    filas = []
    for s in semillas:
        datos = particionar(s)
        for nombre, arq in REFERENCIAS_MANUALES.items():
            test_err, val_err = evaluar_en_prueba(arq, datos, cfg_ent)
            from .arquitectura import contar_parametros
            from .evaluacion import n_salida_mlp
            filas.append({"semilla": s, "referencia": nombre, "arquitectura": str(arq),
                          "n_parametros": contar_parametros(arq, datos.n_entrada, n_salida_mlp(datos.n_clases)),
                          "val_error": val_err, "test_error": test_err, "test_accuracy": 1 - test_err})
    return pd.DataFrame(filas)
