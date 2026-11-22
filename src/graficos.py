"""Figuras científicas del experimento (todas se generan a partir de los CSV reales)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metricas import frente

COLORES = {"NSGA-II": "#1f77b4", "Aleatoria": "#d62728"}
MARCAS = {"NSGA-II": "o", "Aleatoria": "s"}
ETQ_ERR = "Error de validación (1 − accuracy)  ↓ mejor"
ETQ_PAR = "Número de parámetros entrenables (escala log)  ↓ mejor"


def _guardar(fig, ruta):
    if ruta:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return fig


def fig_frente_un_metodo(evals, tag, metodo, semilla, ruta=None):
    """Figura 1/2: archivo completo (gris) y frente no dominado de un método en una corrida."""
    g = evals[(evals.tag == tag) & (evals.metodo == metodo) & (evals.semilla == semilla)]
    f = frente(g).sort_values("n_parametros")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(g.n_parametros, g.val_error, s=18, c="lightgray", label=f"Evaluadas ({len(g)})")
    ax.step(f.n_parametros, f.val_error, where="post", c=COLORES[metodo], lw=1.2)
    ax.scatter(f.n_parametros, f.val_error, s=42, c=COLORES[metodo], marker=MARCAS[metodo],
               label=f"Frente de Pareto ({len(f)})", zorder=3)
    ax.set_xscale("log"); ax.set_xlabel(ETQ_PAR); ax.set_ylabel(ETQ_ERR)
    ax.set_title(f"{metodo}: frente de Pareto de validación (semilla {semilla}, {len(g)} evaluaciones)")
    ax.grid(alpha=.3); ax.legend()
    return _guardar(fig, ruta)


def fig_frentes_conjuntos(evals, tag, semilla, ruta=None):
    """Figura 3: ambos frentes en la misma corrida/partición (comparación pareada)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in ("Aleatoria", "NSGA-II"):
        g = evals[(evals.tag == tag) & (evals.metodo == m) & (evals.semilla == semilla)]
        f = frente(g).sort_values("n_parametros")
        ax.step(f.n_parametros, f.val_error, where="post", c=COLORES[m], lw=1.2, alpha=.8)
        ax.scatter(f.n_parametros, f.val_error, s=44, c=COLORES[m], marker=MARCAS[m],
                   label=f"{m} ({len(f)} no dominadas)", alpha=.85)
    ax.set_xscale("log"); ax.set_xlabel(ETQ_PAR); ax.set_ylabel(ETQ_ERR)
    ax.set_title(f"Frentes de Pareto de validación, misma partición (semilla {semilla})")
    ax.grid(alpha=.3); ax.legend()
    return _guardar(fig, ruta)


def fig_error_vs_parametros(evals, tag, ruta=None):
    """Figura 4: todas las arquitecturas evaluadas (todas las semillas) por método."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, m in zip(axes, ("NSGA-II", "Aleatoria")):
        g = evals[(evals.tag == tag) & (evals.metodo == m)]
        ax.scatter(g.n_parametros, g.val_error, s=8, alpha=.25, c=COLORES[m])
        ax.set_xscale("log"); ax.set_xlabel(ETQ_PAR); ax.set_title(f"{m} (n = {len(g)} evaluaciones)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel(ETQ_ERR)
    fig.suptitle("Error de validación frente a complejidad: todas las arquitecturas evaluadas (10 corridas)", y=1.02)
    return _guardar(fig, ruta)


def fig_distribucion(tabla, tag, columna, titulo, etiqueta, ruta=None):
    """Figura 6: distribución entre ejecuciones (caja + puntos pareados por semilla)."""
    t = tabla[tabla.tag == tag].pivot(index="semilla", columns="metodo", values=columna)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    orden = ["Aleatoria", "NSGA-II"]
    ax.boxplot([t[m] for m in orden], widths=.5, showfliers=False)
    for s in t.index:
        ax.plot([1, 2], [t.loc[s, orden[0]], t.loc[s, orden[1]]], c="gray", alpha=.4, lw=.8)
    for i, m in enumerate(orden, 1):
        ax.scatter(np.full(len(t), i), t[m], c=COLORES[m], s=30, zorder=3)
    ax.set_xticks([1, 2], orden); ax.set_ylabel(etiqueta); ax.set_title(titulo); ax.grid(alpha=.3, axis="y")
    return _guardar(fig, ruta)


def fig_convergencia(curva, tag, ruta=None):
    """HV medio (± desviación) frente al número de evaluaciones consumidas."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in ("Aleatoria", "NSGA-II"):
        c = curva[(curva.tag == tag) & (curva.metodo == m)].groupby("evaluaciones")["hv"].agg(["mean", "std"])
        ax.plot(c.index, c["mean"], marker=MARCAS[m], c=COLORES[m], label=m)
        ax.fill_between(c.index, c["mean"] - c["std"], c["mean"] + c["std"], color=COLORES[m], alpha=.15)
    ax.set_xlabel("Evaluaciones consumidas (arquitecturas entrenadas)")
    ax.set_ylabel("Hipervolumen normalizado  ↑ mejor")
    ax.set_title("Convergencia: hipervolumen del archivo vs. presupuesto (media ± 1 d.e., 10 semillas)")
    ax.grid(alpha=.3); ax.legend()
    return _guardar(fig, ruta)


def fig_mejores_soluciones(sel, tag, ruta=None):
    """Figura 5: soluciones seleccionadas, medidas en prueba (mediana e IQR, 10 corridas).

    Se usan puntos con barras de error (no barras): en escala logarítmica la altura de una
    barra depende de un origen arbitrario y sería engañosa.
    """
    s = sel[sel.tag == tag]
    criterios = ["mejor_error", "codo", "compacta"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    desplazo = {"Aleatoria": -0.12, "NSGA-II": 0.12}
    for m in ("Aleatoria", "NSGA-II"):
        for ax, col in ((axes[0], "test_error"), (axes[1], "n_parametros")):
            med, q1, q3 = [], [], []
            for c in criterios:
                v = s[(s.metodo == m) & (s.criterio == c)][col]
                med.append(v.median()); q1.append(v.quantile(.25)); q3.append(v.quantile(.75))
            med, q1, q3 = map(np.array, (med, q1, q3))
            ax.errorbar(np.arange(len(criterios)) + desplazo[m], med, yerr=[med - q1, q3 - med], fmt=MARCAS[m],
                        color=COLORES[m], capsize=4, ms=8, label=m)
    axes[0].set_ylabel("Error de prueba (1 − accuracy)  ↓ mejor")
    axes[1].set_ylabel("Parámetros entrenables (escala log)  ↓ mejor")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(range(len(criterios)), ["Menor error\nde validación", "Codo", "Compacta\n(≤ +1 pp)"])
        ax.set_xlim(-0.5, 2.5); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("Soluciones elegidas con validación y medidas en prueba (mediana e IQR, 10 corridas)")
    return _guardar(fig, ruta)


def fig_sensibilidad(tabla, columna, etiqueta, ruta=None):
    """Sensibilidad al tamaño de población: columna por N (NSGA-II) vs. búsqueda aleatoria."""
    mapa = {"sens_pob10": 10, "principal": 20, "sens_pob50": 50}
    datos, etiquetas = [], []
    for tag, n in sorted(mapa.items(), key=lambda kv: kv[1]):
        v = tabla[(tabla.tag == tag) & (tabla.metodo == "NSGA-II")][columna]
        if len(v):
            datos.append(v.to_numpy()); etiquetas.append(f"N = {n}")
    ref = tabla[(tabla.tag == "principal") & (tabla.metodo == "Aleatoria")][columna]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.boxplot(datos + [ref.to_numpy()], widths=.5, showfliers=False)
    ax.set_xticks(range(1, len(datos) + 2), etiquetas + ["Aleatoria"])
    ax.axhline(ref.median(), color=COLORES["Aleatoria"], ls="--", lw=1, label="mediana aleatoria")
    ax.set_ylabel(etiqueta); ax.set_xlabel("NSGA-II: tamaño de población (presupuesto fijo = 200)")
    ax.set_title("Sensibilidad al tamaño de población"); ax.grid(alpha=.3, axis="y"); ax.legend()
    return _guardar(fig, ruta)
