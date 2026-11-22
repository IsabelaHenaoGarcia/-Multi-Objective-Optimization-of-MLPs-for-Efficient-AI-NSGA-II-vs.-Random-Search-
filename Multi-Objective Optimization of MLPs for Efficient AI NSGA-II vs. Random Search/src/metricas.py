"""Métricas por corrida, selección final de soluciones y estadística comparativa."""
import numpy as np
import pandas as pd
from scipy import stats

from .arquitectura import Arquitectura, contar_parametros
from .config import ConfigMetricas, EspacioBusqueda
from .datos import particionar
from .evaluacion import evaluar_en_prueba, n_salida_mlp
from .experimento import desde_texto
from .pareto import hipervolumen_2d, mascara_no_dominados

N_ENTRADA, N_SALIDA = 64, 10   # Digits


def extremos_parametros(espacio=EspacioBusqueda()):
    """Parámetros de la arquitectura más pequeña y de la más grande del espacio."""
    pmin = contar_parametros(Arquitectura((espacio.unidades_min,), espacio.activaciones[0]), N_ENTRADA, N_SALIDA)
    pmax = contar_parametros(Arquitectura((espacio.unidades_max,) * espacio.capas_max, espacio.activaciones[0]),
                             N_ENTRADA, N_SALIDA)
    return pmin, pmax


def frente(df):
    """Subconjunto no dominado (error, parámetros) del ARCHIVO de arquitecturas evaluadas."""
    return df[mascara_no_dominados(df[["val_error", "n_parametros"]].to_numpy())]


def hipervolumen_normalizado(df, cfg=ConfigMetricas(), espacio=EspacioBusqueda(), escala="log", error_ref=None):
    """HV en (error, log10 parámetros) normalizado a [0,1] por el rectángulo ideal-referencia.

    Referencia común a todos los métodos y corridas: (error_ref, parámetros de la mayor red).
    El punto ideal es (0, parámetros de la menor red). `escala='lineal'` usa parámetros crudos.
    """
    pmin, pmax = extremos_parametros(espacio)
    e_ref = cfg.error_ref if error_ref is None else error_ref
    if escala == "log":
        F = np.column_stack([df["val_error"], np.log10(df["n_parametros"])])
        ref, ancho = (e_ref, np.log10(pmax)), np.log10(pmax) - np.log10(pmin)
    else:
        F = np.column_stack([df["val_error"], df["n_parametros"]])
        ref, ancho = (e_ref, pmax), pmax - pmin
    return hipervolumen_2d(F, ref) / (e_ref * ancho)


def metricas_corrida(df, tiempo_s, cfg=ConfigMetricas(), espacio=EspacioBusqueda()):
    f = frente(df)
    mejor = df["val_error"].min()
    return {
        "hv": hipervolumen_normalizado(df, cfg, espacio),
        "hv_ref050": hipervolumen_normalizado(df, cfg, espacio, error_ref=0.5),
        "hv_lineal": hipervolumen_normalizado(df, cfg, espacio, escala="lineal"),
        "tam_frente": len(f),
        "mejor_val_error": mejor,
        "min_param_tol": int(df.loc[df["val_error"] <= mejor + cfg.tolerancia_compacta, "n_parametros"].min()),
        "min_param_global": int(df["n_parametros"].min()),
        "tiempo_s": tiempo_s,
        "n_no_convergio": int((~df["convergio"]).sum()),
        "evaluaciones": len(df),
    }


def tabla_metricas(evals, ejecuciones, cfg=ConfigMetricas()):
    filas = []
    for (tag, metodo, semilla), g in evals.groupby(["tag", "metodo", "semilla"]):
        t = ejecuciones[(ejecuciones.tag == tag) & (ejecuciones.metodo == metodo) &
                        (ejecuciones.semilla == semilla)]["tiempo_s"].iloc[0]
        filas.append({"tag": tag, "metodo": metodo, "semilla": semilla, **metricas_corrida(g, t, cfg)})
    return pd.DataFrame(filas)


def curva_hv(evals, puntos, cfg=ConfigMetricas()):
    """HV del archivo tras las primeras b evaluaciones (b en `puntos`), por corrida."""
    filas = []
    for (tag, metodo, semilla), g in evals.groupby(["tag", "metodo", "semilla"]):
        g = g.sort_values("orden")
        for b in puntos:
            filas.append({"tag": tag, "metodo": metodo, "semilla": semilla, "evaluaciones": b,
                          "hv": hipervolumen_normalizado(g[g["orden"] <= b], cfg)})
    return pd.DataFrame(filas)


def resumen_descriptivo(tabla, columnas):
    """Media, desviación estándar (ddof=1), mediana y rango intercuartílico por (tag, método)."""
    filas = []
    for (tag, metodo), g in tabla.groupby(["tag", "metodo"]):
        fila = {"tag": tag, "metodo": metodo, "n": len(g)}
        for c in columnas:
            x = g[c]
            fila[f"{c}_media"] = x.mean()
            fila[f"{c}_de"] = x.std(ddof=1)
            fila[f"{c}_mediana"] = x.median()
            fila[f"{c}_iqr"] = x.quantile(.75) - x.quantile(.25)
        filas.append(fila)
    return pd.DataFrame(filas)


def wilcoxon_pareado_exacto(x, y):
    """Wilcoxon de rangos con signo, bilateral, por permutación exacta (sin aproximaciones).

    Independiente de la versión de scipy (cuyo tratamiento de empates ha cambiado entre versiones):
    descarta diferencias nulas (zero_method='wilcox'), usa rangos medios si hay empates en |d| y
    enumera por programación dinámica todas las 2^n asignaciones de signo, con aritmética entera
    (rangos x 2). p = P(|T+ - E| >= |T+_obs - E|) bajo H0.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    r2 = np.round(2 * stats.rankdata(np.abs(d))).astype(np.int64)   # rangos medios duplicados: enteros
    total = int(r2.sum())
    obs = int(r2[d > 0].sum())
    cuenta = np.zeros(total + 1, dtype=object)                        # enteros de Python: sin desbordes
    cuenta[0] = 1
    for r in r2:
        nuevo = cuenta.copy()
        nuevo[r:] += cuenta[:total + 1 - r]
        cuenta = nuevo
    k = np.arange(total + 1)
    extremo = np.abs(2 * k - total) >= abs(2 * obs - total)
    return float(sum(cuenta[extremo]) / 2 ** len(d))


def comparacion_pareada(tabla, tag, metrica, mayor_es_mejor=True, a="NSGA-II", b="Aleatoria"):
    """Wilcoxon pareado exacto (bilateral) sobre pares con la misma semilla; ver wilcoxon_pareado_exacto."""
    t = tabla[tabla.tag == tag].pivot(index="semilla", columns="metodo", values=metrica).dropna()
    x, y = t[a].to_numpy(), t[b].to_numpy()
    d = x - y
    sentido = 1 if mayor_es_mejor else -1
    victorias = int(np.sum(sentido * d > 0)), int(np.sum(sentido * d < 0)), int(np.sum(d == 0))
    p = wilcoxon_pareado_exacto(x, y)
    return {"metrica": metrica, "n_pares": len(d), f"mediana_{a}": float(np.median(x)),
            f"mediana_{b}": float(np.median(y)), "mediana_diferencia": float(np.median(d)),
            "victorias_NSGA2": victorias[0], "victorias_Aleatoria": victorias[1], "empates": victorias[2],
            "p_wilcoxon_bilateral": p}


def _elegir(f, cfg):
    """Tres criterios de selección final sobre un frente de VALIDACIÓN (sin usar prueba)."""
    f = f.sort_values(["val_error", "n_parametros"]).reset_index(drop=True)
    elegidas = {"mejor_error": f.iloc[0]}
    # Codo: menor distancia euclídea al ideal con ambos objetivos normalizados (min-max del frente)
    rng_e = max(f["val_error"].max() - f["val_error"].min(), 1e-12)
    rng_p = max(f["n_parametros"].max() - f["n_parametros"].min(), 1e-12)
    d = np.hypot((f["val_error"] - f["val_error"].min()) / rng_e,
                 (f["n_parametros"] - f["n_parametros"].min()) / rng_p)
    elegidas["codo"] = f.iloc[int(np.argmin(d.to_numpy()))]
    # Compacta: menos parámetros con error <= mejor + tolerancia (dispositivo limitado)
    ok = f[f["val_error"] <= f["val_error"].min() + cfg.tolerancia_compacta]
    elegidas["compacta"] = ok.sort_values(["n_parametros", "val_error"]).iloc[0]
    return elegidas


def seleccion_final(evals, cfg=ConfigMetricas()):
    """Para cada corrida: elige 3 soluciones con validación y las mide UNA vez en prueba."""
    filas = []
    for (tag, metodo, semilla), g in evals.groupby(["tag", "metodo", "semilla"]):
        datos = particionar(int(semilla))
        for criterio, fila in _elegir(frente(g), cfg).items():
            arq = desde_texto(fila["capas"], fila["activacion"])
            test_err, val_err = evaluar_en_prueba(arq, datos)
            if abs(val_err - fila["val_error"]) > 1e-12:
                raise AssertionError("El reentrenamiento no reproduce el error de validación registrado")
            filas.append({"tag": tag, "metodo": metodo, "semilla": semilla, "criterio": criterio,
                          "arquitectura": str(arq), "n_parametros": int(fila["n_parametros"]),
                          "val_error": val_err, "test_error": test_err, "test_accuracy": 1 - test_err,
                          "brecha_test_menos_val": test_err - val_err})
    return pd.DataFrame(filas)


def cobertura_entre_metodos(evals, tag, a="NSGA-II", b="Aleatoria"):
    """C(a,b) y C(b,a) de Zitzler sobre los frentes de cada semilla (fracción dominada o igualada)."""
    from .pareto import cobertura
    filas = []
    for s, g in evals[evals.tag == tag].groupby("semilla"):
        fa = frente(g[g.metodo == a])[["val_error", "n_parametros"]].to_numpy()
        fb = frente(g[g.metodo == b])[["val_error", "n_parametros"]].to_numpy()
        filas.append({"semilla": s, f"C({a},{b})": cobertura(fa, fb), f"C({b},{a})": cobertura(fb, fa)})
    return pd.DataFrame(filas)


def comparacion_por_presupuesto(curva, tag, a="NSGA-II", b="Aleatoria"):
    """Wilcoxon pareado del HV a distintos presupuestos (prefijos de la misma corrida)."""
    filas = []
    for bud, g in curva[curva.tag == tag].groupby("evaluaciones"):
        t = g.pivot(index="semilla", columns="metodo", values="hv").dropna()
        d = (t[a] - t[b]).to_numpy()
        p = wilcoxon_pareado_exacto(t[a], t[b])
        filas.append({"presupuesto": int(bud), f"mediana_hv_{a}": t[a].median(), f"mediana_hv_{b}": t[b].median(),
                      "mediana_diferencia": float(np.median(d)), "victorias_NSGA2": int((d > 0).sum()),
                      "victorias_Aleatoria": int((d < 0).sum()), "empates": int((d == 0).sum()), "p_wilcoxon": p})
    return pd.DataFrame(filas)
