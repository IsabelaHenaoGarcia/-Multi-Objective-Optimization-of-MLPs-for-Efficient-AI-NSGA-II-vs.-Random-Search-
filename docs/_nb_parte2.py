"""Celdas ejecutables del notebook (secciones 13-20). `I[clave]` son interpretaciones escritas tras ver las salidas."""


def parte2(md, code, I):
    c = []
    c.append(md(r"""## 13. Arquitectura experimental
Estructura del proyecto (todo con rutas relativas):
```
src/            config · datos · arquitectura · evaluacion · pareto · nsga2 · busqueda_aleatoria · experimento · metricas · graficos
tests/          pruebas automáticas (fórmulas, NSGA-II vs fuerza bruta, hipervolumen vs malla, presupuesto, equidad)
resultados/     CSV de cada corrida y tablas finales        figuras/   gráficos generados por este cuaderno
ejecutar_experimentos.py   reproduce todas las corridas (reanudable)
```
**Flujo:** semilla → partición 60/20/20 → (NSGA-II | aleatoria) con un `EvaluadorPresupuestado` que cuenta entrenamientos → archivo de arquitecturas evaluadas → frente no dominado →
métricas → selección final con validación → **única** medición en prueba."""))

    c.append(code(r'''# --- Configuración global y semillas ---------------------------------------------------------
import sys, json, inspect, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

sys.path.insert(0, str(Path.cwd()))          # el cuaderno vive en la raíz del proyecto
if not (Path.cwd() / "src").is_dir():
    raise FileNotFoundError("Ejecute el cuaderno desde la carpeta del proyecto (la que contiene src/). En Colab: %cd /content/nsga2_mlp_computacion_blanda")
from src.config import (EspacioBusqueda, ConfigEntrenamiento, ConfigBusqueda, ConfigDatos,
                        ConfigMetricas, SEMILLAS_POR_DEFECTO)
from src.datos import particionar
from src.arquitectura import Arquitectura, contar_parametros
from src.experimento import (ejecutar_corrida, cargar_resultados, evaluar_referencias_manuales)
from src import metricas as M, graficos as G, nsga2, busqueda_aleatoria, pareto, arquitectura as A, evaluacion

pd.set_option("display.width", 180); pd.set_option("display.max_columns", 30)
MODO = "guardados"     # "guardados": lee resultados/ · "completo": vuelve a ejecutar todo (~30 min en 1 CPU)
DIR_RES, DIR_FIG = Path("resultados"), Path("figuras")
SEMILLAS = list(SEMILLAS_POR_DEFECTO)
if MODO == "completo":
    subprocess.run([sys.executable, "ejecutar_experimentos.py", "--modo", "completo", "--rehacer"], check=True)
if not (DIR_RES / "corridas").exists():
    raise FileNotFoundError("Faltan resultados/: ejecute `python ejecutar_experimentos.py` o ponga MODO='completo'.")
print("Espacio:", EspacioBusqueda()); print("Entrenamiento:", ConfigEntrenamiento())
print("Búsqueda (principal):", ConfigBusqueda()); print("Métricas:", ConfigMetricas())
print("Entorno usado en las corridas:", json.loads((DIR_RES / "entorno.json").read_text())["scikit-learn"], "(scikit-learn)")'''))

    c.append(code(r'''# --- Datos: tamaños de partición y comprobaciones anti-fuga ------------------------------------
d = particionar(0)
print(f"Entrenamiento {d.X_ent.shape}, validación {d.X_val.shape}, prueba {d.X_prueba.shape}; clases = {d.n_clases}")
print("Media del escalador en entrenamiento ≈ 0 :", bool(np.allclose(d.X_ent.mean(0), 0, atol=1e-8)),
      "| en validación ≈ 0 :", bool(np.allclose(d.X_val.mean(0), 0, atol=1e-8)), "(debe ser False: solo se ajusta con entrenamiento)")
print("Proporción por clase en entrenamiento:", np.round(np.bincount(d.y_ent) / len(d.y_ent), 3))
pmin, pmax = M.extremos_parametros()
print(f"Parámetros de la red más pequeña del espacio: {pmin} · de la más grande: {pmax}")
tamano_espacio = sum((EspacioBusqueda().unidades_max - EspacioBusqueda().unidades_min + 1) ** L for L in (1, 2, 3)) * 2
print(f"Tamaño del espacio de arquitecturas (fenotipos): {tamano_espacio:,}")'''))

    c.append(md(r"""## 14. Implementación de NSGA-II
Se implementó **desde cero** (sin DEAP) para poder explicar cada operador. Decisiones principales:
* **Codificación:** genoma `(n_capas, u1, u2, u3, activación)`; solo las primeras `n_capas` unidades forman el fenotipo. Decodificar valida rangos; cruce y mutación recortan al rango, por lo que **no generan arquitecturas inválidas**.
* **Cruce:** uniforme (cada gen de un padre al azar). **Mutación:** por gen con $p=1/5$; capas ±1; unidades × $e^{\mathcal N(0,0{,}6)}$; activación alternada.
* **Sin duplicados:** si un hijo ya fue evaluado se reintenta (hasta 100 veces); si fallara se usa una muestra aleatoria única y se **contabiliza** (`hijos_fallback_aleatorio`; ver sección 17).
* **Presupuesto:** `validar_presupuesto_nsga2` exige que $(B-N)$ sea múltiplo de $N$; el `EvaluadorPresupuestado` lanza error si se intenta evaluar de más.

Código clave (se muestra el que realmente se ejecutó):"""))
    c.append(code(r'''for f in (pareto.ordenamiento_no_dominado, pareto.distancia_crowding, pareto.seleccionar_supervivientes):
    print(inspect.getsource(f))'''))
    c.append(code(r'''for f in (nsga2._torneo_binario, nsga2._generar_hijo, nsga2.ejecutar_nsga2):
    print(inspect.getsource(f))'''))
    c.append(md(r"""**Verificación.** Las piezas matemáticas se comprobaron contra implementaciones independientes: ordenamiento no dominado vs fuerza bruta (20 conjuntos aleatorios con empates), hipervolumen vs integración por malla fina (20 conjuntos),
crowding y supervivientes con ejemplos calculados a mano, fórmula de parámetros contra los tensores de scikit-learn, y propiedades de presupuesto, duplicados y reproducibilidad. La siguiente celda las ejecuta:"""))
    c.append(code(r'''import importlib.util
if importlib.util.find_spec("pytest") is None:
    print("pytest no está instalado: omita esta celda o ejecute `pip install pytest` (las pruebas están en tests/).")
else:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1])
    assert r.returncode == 0, r.stdout[-2000:]'''))

    c.append(md(r"""## 15. Implementación de búsqueda aleatoria
Usa el **mismo** espacio, el **mismo** muestreador (logarítmico en neuronas) y el **mismo** evaluador con presupuesto. Comparte el flujo aleatorio con la población inicial de NSGA-II,
de modo que sus primeras $N=20$ arquitecturas coinciden (diseño pareado)."""))
    c.append(code(r'''print(inspect.getsource(busqueda_aleatoria.ejecutar_busqueda_aleatoria))'''))

    c.append(md(r"""## 16. Ejecución de experimentos
**Protocolo ejecutado** (`ejecutar_experimentos.py`): 10 semillas (0–9) × 2 métodos × 200 evaluaciones (conjunto *principal*), más NSGA-II con $N=10$ y $N=50$ (sensibilidad, 10 semillas, 200 evaluaciones).
Cada corrida se guarda en `resultados/corridas/*.csv` junto con su `.json` (tiempo, configuración, contadores). El entorno (versiones, semillas) está en `resultados/entorno.json`.

**Demostración en vivo** (presupuesto 40, $N=10$, semilla 0; unos 15 s) y **comprobación de reproducibilidad**: NSGA-II con $N=20$ es secuencial, así que una corrida de presupuesto 60 debe coincidir exactamente con las primeras 60 evaluaciones de la corrida guardada de presupuesto 200."""))
    c.append(code(r'''# Demostración en vivo
cfg_demo = ConfigBusqueda(presupuesto=40, tam_poblacion=10)
for m in ("NSGA-II", "Aleatoria"):
    df, res = ejecutar_corrida(m, 0, cfg_demo)
    f = M.frente(df).sort_values("n_parametros")
    print(f"{m:9s}: {len(df)} evaluaciones reales, {res.tiempo_s:.1f} s, frente de {len(f)} arquitecturas, mejor error {df.val_error.min():.4f}")
display(f[["capas", "activacion", "n_parametros", "val_error"]].reset_index(drop=True).head(6))'''))
    c.append(code(r'''# Reproducibilidad: prefijo de 60 evaluaciones de NSGA-II (N=20) frente a la corrida guardada
guardada = pd.read_csv(DIR_RES / "corridas" / "principal__NSGA-II__s0.csv").sort_values("orden").head(60).reset_index(drop=True)
nueva, _ = ejecutar_corrida("NSGA-II", 0, ConfigBusqueda(presupuesto=60, tam_poblacion=20))
nueva = nueva.sort_values("orden").reset_index(drop=True)
iguales = (guardada[["capas", "activacion"]].equals(nueva[["capas", "activacion"]])
           and np.allclose(guardada.val_error, nueva.val_error, atol=0) and (guardada.n_parametros == nueva.n_parametros).all())
print("Reproducción idéntica de las primeras 60 evaluaciones:", iguales)
assert iguales'''))

    c.append(md(r"""## 17. Resultados
Se leen todas las corridas guardadas. Primero, **comprobaciones de integridad** del diseño experimental; solo después se interpretan resultados."""))
    c.append(code(r'''ev_all, ej_all = cargar_resultados(DIR_RES)
ev, ej = ev_all[ev_all.tag == "principal"], ej_all[ej_all.tag == "principal"]

# Integridad: presupuesto, duplicados, finitud, equidad de datos
chequeos = []
for (tag, m, s), g in ev_all.groupby(["tag", "metodo", "semilla"]):
    esperado = {"principal": 200, "sens_pob10": 200, "sens_pob50": 200}[tag]
    assert len(g) == esperado, (tag, m, s, len(g))
    assert g.duplicated(["capas", "activacion"]).sum() == 0, "arquitectura duplicada"
    assert np.isfinite(g[["val_error", "n_parametros"]].to_numpy()).all()
    assert g.n_parametros.min() >= pmin and g.n_parametros.max() <= pmax
print(f"{ev_all.groupby(['tag','metodo','semilla']).ngroups} corridas: todas con 200 evaluaciones reales, sin duplicados, sin valores no finitos, dentro de los límites.")
print("Hijos reemplazados por muestra aleatoria (fallback) en NSGA-II:", int(ej_all[ej_all.metodo == 'NSGA-II'].fallback_aleatorio.sum()))
pareadas = ev.pivot_table(index="semilla", columns="metodo", values="orden", aggfunc="count")
print("Evaluaciones por (semilla, método) del conjunto principal:\n", pareadas.T.to_string())
n_ini = ev[ev.orden <= 20].sort_values(["semilla", "orden"])
mismo = all(n_ini[(n_ini.semilla == s) & (n_ini.metodo == "NSGA-II")][["capas", "activacion"]].reset_index(drop=True)
            .equals(n_ini[(n_ini.semilla == s) & (n_ini.metodo == "Aleatoria")][["capas", "activacion"]].reset_index(drop=True)) for s in SEMILLAS)
print("Primeras 20 evaluaciones idénticas en ambos métodos (diseño pareado):", mismo)
print(f"Entrenamientos que NO convergieron en 100 épocas: aleatoria {1-ev[ev.metodo=='Aleatoria'].convergio.mean():.1%}, NSGA-II {1-ev[ev.metodo=='NSGA-II'].convergio.mean():.1%}")'''))
    c.append(code(r'''tabla = M.tabla_metricas(ev_all, ej_all)
tabla.to_csv(DIR_RES / "metricas_por_ejecucion.csv", index=False)
COLS = ["hv", "tam_frente", "mejor_val_error", "min_param_tol", "tiempo_s"]
resumen = M.resumen_descriptivo(tabla, COLS)
resumen.to_csv(DIR_RES / "resumen_metricas.csv", index=False)
p = resumen[resumen.tag == "principal"].set_index("metodo")
filas = []
for c_, nombre in [("hv", "Hipervolumen normalizado (↑)"), ("tam_frente", "Tamaño del frente (↑)"),
                   ("mejor_val_error", "Mejor error de validación (↓)"), ("min_param_tol", "Parámetros mínimos con error ≤ mejor+1 pp (↓)"),
                   ("tiempo_s", "Tiempo de búsqueda, s (informativo)")]:
    for m in ("NSGA-II", "Aleatoria"):
        filas.append({"métrica": nombre, "método": m, "media": p.loc[m, f"{c_}_media"], "d.e.": p.loc[m, f"{c_}_de"],
                      "mediana": p.loc[m, f"{c_}_mediana"], "IQR": p.loc[m, f"{c_}_iqr"]})
display(pd.DataFrame(filas).round(4).set_index(["métrica", "método"]))'''))
    c.append(md(I["resumen"]))

    c.append(md(r"""## 18. Visualizaciones
Las figuras de una sola corrida muestran la **semilla 0** (la primera, elegida antes de ver resultados; no se seleccionó por ser favorable). Las figuras agregadas usan las 10 semillas."""))
    c.append(code(r'''G.fig_frente_un_metodo(ev, "principal", "NSGA-II", 0, DIR_FIG / "fig1_frente_nsga2.png"); plt.show()
G.fig_frente_un_metodo(ev, "principal", "Aleatoria", 0, DIR_FIG / "fig2_frente_aleatoria.png"); plt.show()
G.fig_frentes_conjuntos(ev, "principal", 0, DIR_FIG / "fig3_frentes_conjuntos.png"); plt.show()'''))
    c.append(md(I["fig123"]))
    c.append(code(r'''G.fig_error_vs_parametros(ev, "principal", DIR_FIG / "fig4_error_vs_parametros.png"); plt.show()'''))
    c.append(md(I["fig4"]))
    c.append(code(r'''G.fig_distribucion(tabla, "principal", "hv", "Hipervolumen normalizado por corrida (líneas = misma semilla)",
                   "Hipervolumen normalizado  ↑ mejor", DIR_FIG / "fig6_distribucion_hv.png"); plt.show()
curva = M.curva_hv(ev_all, [20, 40, 60, 100, 140, 200]); curva.to_csv(DIR_RES / "curva_hv.csv", index=False)
G.fig_convergencia(curva, "principal", DIR_FIG / "fig7_convergencia.png"); plt.show()'''))
    c.append(md(I["fig6"]))

    c.append(md(r"""## 19. Análisis estadístico
**Hipervolumen.** Área del espacio (error, log₁₀ parámetros) dominada por el frente y acotada por el punto de referencia **común** $(0{,}9;\ P_{\max})$, dividida por el área del rectángulo ideal–referencia, de modo que está en $[0,1]$. El error de referencia 0,9 es el de adivinar en 10 clases balanceadas;
$P_{\max}$ es la red mayor del espacio. El logaritmo refleja que pasar de 200 a 400 parámetros pesa más que de 12 000 a 12 200. Se calcula sobre el **archivo** (todo lo evaluado) de cada método, igual para ambos.
**Robustez a la elección de la métrica:** también se reporta con referencia de error 0,5 y con parámetros en escala lineal.

**Prueba.** Wilcoxon de rangos con signo (exacto, bilateral, por permutación; implementación propia validada contra scipy en ausencia de empates, que no depende de la versión de scipy: descarta diferencias nulas y usa rangos medios si hay empates) sobre los 10 pares con igual semilla; se informan victorias por método y la mediana de las diferencias. Con $n=10$ la potencia es baja: un $p$ alto **no demuestra equivalencia**.
Se usa $\alpha=0{,}05$ fijado antes de ver los resultados; las seis comparaciones siguientes son exploratorias y no se corrigen por múltiples pruebas (hacerlo solo reduciría aún más las diferencias significativas)."""))
    c.append(code(r'''mets = [("hv", True), ("hv_ref050", True), ("hv_lineal", True), ("tam_frente", True), ("mejor_val_error", False), ("min_param_tol", False)]
comp = pd.DataFrame([M.comparacion_pareada(tabla, "principal", m, mayor) for m, mayor in mets])
comp.to_csv(DIR_RES / "comparaciones_wilcoxon.csv", index=False)
display(comp.round(4).set_index("metrica"))
cob = M.cobertura_entre_metodos(ev, "principal"); cob.to_csv(DIR_RES / "cobertura_C.csv", index=False)
print("Cobertura C de Zitzler (fracción del frente del otro método que es dominada o igualada; descriptiva):")
display(cob.drop(columns="semilla").agg(["mean", "median"]).round(3))'''))
    c.append(md(I["stats"]))

    c.append(md(r"""### Evaluación final de las soluciones seleccionadas (única vez que se usa el conjunto de prueba)
Para cada corrida se extrae el frente de **validación** y se eligen tres soluciones con criterios fijados antes (menor error, codo, compacta ≤ +1 pp). Solo entonces se reentrena la arquitectura con la misma semilla (el código verifica que reproduce exactamente el error de validación guardado) y se mide en **prueba**.
Como referencia de contexto, se evalúan tres MLP manuales con el mismo procedimiento."""))
    c.append(code(r'''sel = M.seleccion_final(ev); sel.to_csv(DIR_RES / "seleccion_final.csv", index=False)
orden_c = ["mejor_error", "codo", "compacta"]
res_sel = (sel.groupby(["criterio", "metodo"])[["n_parametros", "val_error", "test_error", "brecha_test_menos_val"]]
           .agg(["median", "mean", "std"]).round(4).loc[orden_c])
display(res_sel)
rows = []
for crit in orden_c:
    for met, mayor in (("test_error", False), ("n_parametros", False)):
        r_ = M.comparacion_pareada(sel[sel.criterio == crit], "principal", met, mayor); r_["criterio"] = crit; rows.append(r_)
comp_sel = pd.DataFrame(rows)[["criterio", "metrica", "mediana_NSGA-II", "mediana_Aleatoria", "victorias_NSGA2", "victorias_Aleatoria", "empates", "p_wilcoxon_bilateral"]]
comp_sel.to_csv(DIR_RES / "comparaciones_seleccion_final.csv", index=False); display(comp_sel.round(4))
ref = pd.read_csv(DIR_RES / "referencias_manuales.csv")
display(ref.groupby("referencia")[["n_parametros", "val_error", "test_error"]].agg(["median"]).round(4))
G.fig_mejores_soluciones(sel, "principal", DIR_FIG / "fig5_mejores_soluciones.png"); plt.show()'''))
    c.append(md(I["test"]))

    c.append(md(r"""## 20. Análisis de sensibilidad (acotado)
Se varía **un factor a la vez**, manteniendo las demás condiciones constantes:
1. **Tamaño de población** de NSGA-II con presupuesto fijo de 200: $N\in\{10,20,50\}$ (19, 9 y 3 generaciones). La búsqueda aleatoria no depende de $N$.
2. **Presupuesto** $B\in\{20,40,60,100,140,200\}$ evaluaciones. Como NSGA-II ($N=20$) y la aleatoria son secuenciales con la misma semilla, el resultado con presupuesto $B$ es exactamente el prefijo de $B$ evaluaciones de la corrida de 200 (comprobado en la sección 16), así que no requiere reentrenar.
Las mismas 10 semillas en todos los casos."""))
    c.append(code(r'''mapa = {"sens_pob10": 10, "principal": 20, "sens_pob50": 50}
sp = M.resumen_descriptivo(tabla[tabla.metodo == "NSGA-II"], ["hv", "tam_frente", "min_param_tol"])
sp["N"] = sp.tag.map(mapa); sp = sp.sort_values("N")
alea = tabla[(tabla.tag == "principal") & (tabla.metodo == "Aleatoria")]
print(f"Referencia: búsqueda aleatoria hv mediana {alea.hv.median():.4f} (media {alea.hv.mean():.4f}, d.e. {alea.hv.std():.4f})")
display(sp[["N", "hv_media", "hv_de", "hv_mediana", "hv_iqr", "tam_frente_mediana", "min_param_tol_mediana"]].round(4).set_index("N"))
# Cada N frente a la aleatoria (mismas semillas)
filas = []
for tag, n in mapa.items():
    t = pd.concat([tabla[(tabla.tag == tag) & (tabla.metodo == "NSGA-II")],
                   tabla[(tabla.tag == "principal") & (tabla.metodo == "Aleatoria")].assign(tag=tag)])
    r_ = M.comparacion_pareada(t, tag, "hv"); r_["N"] = n; filas.append(r_)
sens_pob = pd.DataFrame(filas).sort_values("N")[["N", "mediana_NSGA-II", "mediana_Aleatoria", "mediana_diferencia", "victorias_NSGA2", "victorias_Aleatoria", "p_wilcoxon_bilateral"]]
sens_pob.to_csv(DIR_RES / "sensibilidad_poblacion.csv", index=False); display(sens_pob.round(4).set_index("N"))
G.fig_sensibilidad(tabla, "hv", "Hipervolumen normalizado  ↑ mejor", DIR_FIG / "fig8_sensibilidad_poblacion.png"); plt.show()
sens_pres = M.comparacion_por_presupuesto(curva, "principal"); sens_pres.to_csv(DIR_RES / "sensibilidad_presupuesto.csv", index=False)
display(sens_pres.round(4).set_index("presupuesto"))'''))
    c.append(md(I["sens"]))

    c.append(code(r'''# Exportación del archivo y de los frentes de cada corrida
ev_all.to_csv(DIR_RES / "evaluaciones_todas.csv", index=False)
frentes = pd.concat([M.frente(g).assign(tag=t, metodo=m, semilla=s) for (t, m, s), g in ev_all.groupby(["tag", "metodo", "semilla"])])
frentes.sort_values(["tag", "metodo", "semilla", "n_parametros"]).to_csv(DIR_RES / "frentes_pareto.csv", index=False)
# mediciones.csv con el esquema de la guía del docente (metrica_principal = hipervolumen normalizado, mismo significado siempre)
campos = ["ejecucion", "metodo", "instancia", "semilla", "metrica_principal", "tiempo_s", "evaluaciones", "observaciones"]
med = []
for i, (_, r) in enumerate(tabla.sort_values(["tag", "metodo", "semilla"]).iterrows(), 1):
    assert np.isfinite(r.hv), "ejecución con métrica no finita: registrarla aparte"
    med.append({"ejecucion": i, "metodo": r.metodo, "instancia": "digits" if r.tag == "principal" else f"digits_{r.tag}",
                "semilla": int(r.semilla), "metrica_principal": r.hv, "tiempo_s": round(r.tiempo_s, 2),
                "evaluaciones": int(r.evaluaciones),
                "observaciones": f"HV normalizado (ref error 0.9, P_max, log10 parametros); {int(r.n_no_convergio)} entrenamientos sin converger en 100 epocas"})
pd.DataFrame(med, columns=campos).to_csv(DIR_RES / "mediciones.csv", index=False)
print("Archivos en resultados/:", sorted(p.name for p in DIR_RES.glob("*.csv")))
print("Figuras en figuras/:", sorted(p.name for p in DIR_FIG.glob("*.png")))'''))
    return c
