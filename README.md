# Optimización multiobjetivo de MLP para IA eficiente: NSGA-II vs. búsqueda aleatoria

Compara NSGA-II (implementado desde cero) con una búsqueda
aleatoria, ambos con **200 evaluaciones** (200 redes entrenadas), para encontrar MLP pequeñas que equilibren
**error de validación** y **número de parámetros** en el conjunto *Digits* (scikit-learn, 1797×64, CC BY 4.0).

## Contenido
| Ruta | Qué es |
|---|---|
| `Investigacion_Tema05.ipynb` | Cuaderno completo y **ya ejecutad**(renombrar con sus apellidos) |
| `src/` | Código: `config`, `datos`, `arquitectura`, `evaluacion`, `pareto`, `nsga2`, `busqueda_aleatoria`, `experimento`, `metricas`, `graficos` |
| `tests/` | 60 pruebas (NSGA-II y métricas vs. fuerza bruta, fórmula de parámetros vs. scikit-learn, presupuesto, equidad) |
| `ejecutar_experimentos.py` | Reproduce todas las corridas (reanudable) |
| `resultados/` | CSV de resultados: `mediciones.csv` (esquema del docente), `evaluaciones_todas.csv`, `frentes_pareto.csv`, `metricas_por_ejecucion.csv`, `resumen_metricas.csv`, `comparaciones_*.csv`, `seleccion_final.csv`, `sensibilidad_*.csv`, `entorno.json`, `corridas/` |
| `figuras/` | Figuras 1–8 generadas por el cuaderno |
| `docs/` | `auditoria_rubrica.md` y las partes `_nb_*.py` con las que `construir_notebook.py` regenera el cuaderno. El informe, la presentación, el guion y el banco de preguntas se entregan aparte como documentos en línea (exportables a Word/PDF) |

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt
```
Probado con Python 3.12 (versiones exactas en `requirements.txt` y `resultados/entorno.json`).

## Ejecución
```bash
python -m pytest -q tests                              
python ejecutar_experimentos.py --modo rapido          
python ejecutar_experimentos.py --modo completo        
jupyter nbconvert --to notebook --execute Investigacion_Tema05_Apellido1_Apellido2.ipynb   
```
El cuaderno lee `resultados/` (`MODO = "guardados"`). Con `MODO = "completo"` vuelve a entrenar todo. La ejecución completa es **reanudable**: las corridas ya guardadas se omiten.
Todo es determinista dado el entorno: la semilla fija la partición, la inicialización de pesos y el algoritmo.

## Diseño en una línea
Objetivos (minimizar): error de validación y parámetros entrenables. Espacio: 1–3 capas, 2–64 neuronas, ReLU/tanh. Particiones 60/20/20 por semilla; **la prueba se usa una sola vez**, al final.
10 semillas pareadas; las primeras 20 evaluaciones de ambos métodos coinciden. Métrica principal: hipervolumen normalizado (referencia común). Prueba: Wilcoxon pareado.

## Resultado principal (de `resultados/`, ver el cuaderno para el análisis)
Hipervolumen mediano 0,9198 (NSGA-II) frente a 0,9111 (aleatoria); NSGA-II mejor en 8 de 10 pares, p = 0,106: **tendencia favorable, no concluyente**. Con parámetros en escala lineal no hay ventaja.
**Límites:** un solo conjunto de datos fácil; 10 semillas; el número de parámetros **no mide energía ni latencia** (no se midieron); 50–54 % de los entrenamientos no convergieron en 100 épocas.
