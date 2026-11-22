"""Celdas del notebook que NO dependen de los resultados (portada, teoría, diseño)."""


def parte1(md, code):
    c = []
    c.append(md(r"""# Optimización multiobjetivo de arquitecturas MLP para IA eficiente
### NSGA-II frente a búsqueda aleatoria con el mismo presupuesto de evaluaciones
**Curso:** Computación Blanda · **Tema 5:** Optimización multiobjetivo para IA eficiente · **Modalidad:** pareja

| Campo | Respuesta |
|---|---|
| Integrante 1 y código | *(completar)* |
| Integrante 2 y código | *(completar)* |
| Grupo del curso | *(completar)* |
| Tema principal y alternativa | Tema 5 (principal) · *(completar alternativa)* |
| Fecha de exposición | *(completar)* |

> **Cómo leer este cuaderno.** Cada afirmación empírica proviene de una celda ejecutable; los números del texto se
> imprimen desde los CSV de `resultados/`. Se distingue siempre entre **resultado** (lo medido), **interpretación**
> (lo que creemos que significa) e **hipótesis** (lo que aún no se ha comprobado).
"""))

    c.append(md(r"""## 2. Resumen
Se evaluó si NSGA-II, implementado desde cero, encuentra mejores compromisos entre **error de validación** y **número de parámetros** que una **búsqueda aleatoria con el mismo presupuesto** (200 redes entrenadas por método) en perceptrones multicapa pequeños sobre el conjunto *Digits* (1797 muestras, 64 variables, 10 clases).
Se hicieron 10 repeticiones pareadas (misma semilla y partición 60/20/20), sin usar el conjunto de prueba durante la optimización, y se compararon con el hipervolumen normalizado de lo evaluado, con una referencia común, y con la prueba de Wilcoxon pareada exacta.

**Resultado principal.** NSGA-II tuvo mayor hipervolumen en 8 de 10 pares (mediana 0,9198 frente a 0,9111), pero $p=0{,}106$: la hipótesis **no queda confirmada**; es una tendencia favorable, no concluyente. Con parámetros en escala lineal no hay ventaja (5 a 5). El compromiso error–tamaño sí es real: reducir los parámetros de ≈ 2400 a ≈ 1250 costó ≈ 0,8 puntos porcentuales de error de prueba.
**Alcance:** un solo conjunto de datos fácil, 10 semillas y entrenamiento corto (50–54 % de los entrenamientos no convergieron en 100 épocas). El número de parámetros es una medida estructural: **no se midió energía ni latencia**.
**Palabras clave:** optimización multiobjetivo, NSGA-II, frente de Pareto, redes neuronales eficientes, búsqueda aleatoria, hipervolumen.
"""))

    c.append(md(r"""## 3. Introducción
Los modelos de aprendizaje automático suelen compararse por su desempeño predictivo, pero desplegarlos exige también
recursos: memoria, tiempo de cómputo y energía. Una red neuronal más grande no siempre es mejor: a partir de cierto
tamaño la mejora es marginal y el costo sigue creciendo. Elegir la arquitectura es, por tanto, un problema con
**objetivos en conflicto**, y la computación evolutiva multiobjetivo es una herramienta natural para estudiarlo porque
devuelve *un conjunto* de compromisos (el frente de Pareto) en lugar de una única "mejor" solución.

## 4. Planteamiento del problema
Dado un conjunto de datos de clasificación y un espacio de arquitecturas MLP pequeñas, ¿cómo encontrar, con un número
limitado de entrenamientos, arquitecturas que ofrezcan buenos compromisos entre **error de validación** y **número de
parámetros entrenables**? Cada evaluación implica entrenar una red, de modo que el presupuesto de evaluaciones es el
recurso escaso que debe igualarse al comparar métodos.

## 5. Justificación
* Conecta **computación evolutiva** (NSGA-II) con **redes neuronales** (MLP), dos pilares de la computación blanda.
* El resultado es útil para decidir arquitecturas en dispositivos con memoria limitada.
* Aporta un protocolo de comparación justa (mismo presupuesto, mismas particiones, diseño pareado) reproducible en un
  computador convencional.

## 6. Pregunta de investigación
**¿Qué tan eficaz es NSGA-II para encontrar arquitecturas de redes neuronales pequeñas que equilibren el error predictivo
y el número de parámetros, en comparación con una búsqueda aleatoria que usa el mismo presupuesto de evaluaciones?**

## 7. Hipótesis
**H1.** NSGA-II encuentra un conjunto de arquitecturas no dominadas que representa mejor el compromiso error–complejidad
que una búsqueda aleatoria con el mismo presupuesto.
*Operacionalización:* mayor hipervolumen normalizado (referencia común) del archivo de arquitecturas evaluadas, comparado
con la prueba de Wilcoxon de rangos con signo sobre pares con la misma semilla. **H0:** no hay diferencia sistemática.
La hipótesis **no se asume verdadera**: puede rechazarse y el trabajo seguiría siendo válido.

## 8. Objetivos
**General.** Evaluar experimentalmente si NSGA-II supera a la búsqueda aleatoria, con igual presupuesto, para hallar
compromisos error–parámetros en MLP pequeñas.

**Específicos.**
1. Formular la búsqueda de arquitecturas como un problema de dos objetivos con una codificación válida por construcción.
2. Implementar y verificar NSGA-II desde cero (ordenamiento no dominado, crowding, torneo, cruce, mutación, elitismo).
3. Implementar una búsqueda aleatoria con el mismo espacio, muestreador y presupuesto.
4. Ejecutar 10 repeticiones pareadas por método, sin usar el conjunto de prueba durante la optimización.
5. Comparar con hipervolumen, tamaño de frente, mejor error y parámetros mínimos, con dispersión y prueba estadística.
6. Realizar un análisis de sensibilidad acotado (tamaño de población y presupuesto) y documentar las limitaciones.
"""))

    c.append(md(r"""## 9. Fundamentos teóricos

### 9.1 Inteligencia artificial eficiente
*Eficiencia* significa obtener un desempeño dado consumiendo menos recursos. Conviene distinguir tres clases, porque **no son equivalentes**:

| Tipo | Qué mide | Ejemplos de indicador |
|---|---|---|
| Estructural | Tamaño del modelo | nº de parámetros, memoria |
| Temporal | Tiempo de cómputo | latencia de inferencia, tiempo de entrenamiento |
| Energética | Energía consumida | julios por inferencia (requiere un medidor) |

Reducir el tamaño importa en dispositivos con memoria, batería o cómputo limitados (sensores, móviles, microcontroladores).
**Advertencia que respetamos en todo el trabajo:** el número de parámetros es una medida *estructural*; no es una medición de
tiempo ni de energía, y aquí **no se mide energía**, por lo que no afirmaremos ahorro energético.

### 9.2 Optimización multiobjetivo
En un problema **monoobjetivo** se minimiza una sola función $f(x)$ y existe (en general) un óptimo. En uno **multiobjetivo**
se minimiza un vector $\mathbf f(x)=(f_1(x),\dots,f_m(x))$ y los objetivos suelen estar en conflicto, así que no hay una
solución que sea mejor en todo. Elementos: *espacio de soluciones* $\mathcal X$ (aquí, las arquitecturas permitidas),
*funciones objetivo* ($f_1$ = error de validación, $f_2$ = parámetros) y *restricciones* (aquí, límites de capas y neuronas).

**Dominancia de Pareto (minimización).** $a$ domina a $b$, escrito $a\prec b$, si $f_i(a)\le f_i(b)$ para todo $i$ y
$f_j(a)<f_j(b)$ para al menos un $j$. Una solución es **no dominada** si ninguna otra la domina; el conjunto de las no dominadas es
el **frente de Pareto** (de lo evaluado, o el verdadero si se conoce el óptimo global).

**Ejemplo numérico.** Sean A = (error 0,04; 600 parámetros), B = (0,05; 900), C = (0,03; 1500), D = (0,04; 600):
* A ≺ B: tiene menos error **y** menos parámetros → B está *dominada*.
* A y C **no se dominan**: C tiene menos error pero más parámetros → ambas son *no dominadas* (representan compromisos distintos).
* A y D son iguales: no hay dominancia entre ellas.

**Elegir una solución final.** El frente ofrece opciones, no una respuesta; la elección depende de una preferencia externa. En este trabajo
se usan tres criterios definidos *a priori* y sobre **validación**: (i) *menor error*, (ii) *codo* (la más cercana al ideal tras normalizar
ambos objetivos) y (iii) *compacta* (la de menos parámetros con error ≤ mejor + 1 punto porcentual), que corresponde al caso de un dispositivo limitado.

### 9.3 Redes neuronales MLP
Una **MLP** (perceptrón multicapa) es una red de capas densas: entrada → capas ocultas → salida. Cada neurona calcula
$a=\phi(\mathbf w^\top\mathbf x+b)$ con **pesos** $\mathbf w$, **sesgo** $b$ y **función de activación** $\phi$ (ReLU $=\max(0,z)$; tanh).
Se entrena minimizando una pérdida (entropía cruzada) con gradiente (Adam); se **valida** en datos no usados para ajustar pesos.

**Parámetros entrenables.** Con $n_0$ entradas, capas ocultas $n_1,\dots,n_L$ y $n_{L+1}$ salidas:
$$P=\sum_{l=1}^{L+1}(n_{l-1}+1)\,n_l .$$
Ejemplo (Digits: $n_0=64$, 10 clases) con una capa de 16 neuronas: $(64+1)\cdot16+(16+1)\cdot10=1040+170=\mathbf{1210}$.
Con 8 y 4 neuronas: $65\cdot8+9\cdot4+5\cdot10=520+36+50=\mathbf{606}$. (Estas cifras se verifican con pruebas automáticas contra scikit-learn.)
La arquitectura determina la complejidad: los parámetros crecen aproximadamente con el producto de neuronas de capas consecutivas.
"""))

    c.append(md(r"""### 9.4 Algoritmo NSGA-II
**Origen.** Propuesto por Deb, Pratap, Agarwal y Meyarivan (2002, *IEEE Trans. Evol. Comput.* 6(2), 182–197) para corregir tres críticas a NSGA:
complejidad $O(MN^3)$, falta de elitismo y necesidad de un parámetro de compartición. NSGA-II ofrece ordenamiento no dominado rápido $O(MN^2)$,
elitismo por combinación padres+hijos y la **distancia de crowding** para preservar diversidad sin parámetros.

**Funcionamiento paso a paso**
1. *Inicialización:* población $P_0$ de $N$ individuos válidos; se evalúan los objetivos (aquí, se entrena cada red).
2. *Ordenamiento no dominado:* se parte la población en frentes $F_0,F_1,\dots$ ($F_0$ = no dominadas).
3. *Crowding:* dentro de cada frente, a cada individuo se le asigna una medida de lo "aislado" que está; los extremos reciben $\infty$.
4. *Selección por torneo binario:* se eligen 2 individuos al azar; gana el de menor rango y, si empatan, el de mayor crowding.
5. *Cruce y mutación:* los padres seleccionados producen hijos $Q_t$ ($N$ hijos).
6. *Reemplazo elitista:* se unen $R_t=P_t\cup Q_t$ ($2N$), se ordena y se llenan $N$ plazas frente por frente; el último frente que no cabe se corta por mayor crowding.
7. *Parada:* tras $G$ generaciones (aquí, hasta agotar el presupuesto de evaluaciones).

**Pseudocódigo**
```
P0 <- N individuos aleatorios válidos ; evaluar(P0)                  # N evaluaciones
for t = 0 .. G-1:
    rango, crowding <- ordenar_no_dominado(P_t), crowding(P_t)
    Q_t <- {}
    while |Q_t| < N:
        p1, p2 <- torneo_binario(P_t), torneo_binario(P_t)
        h <- mutar(cruzar(p1, p2))     # se reintenta si el hijo ya fue evaluado
        Q_t <- Q_t ∪ {h}
    evaluar(Q_t)                                                       # N evaluaciones
    R_t <- P_t ∪ Q_t
    P_{t+1} <- primeros N de R_t por (rango asc., crowding desc.)
devolver el frente no dominado de lo evaluado
```
**Contabilidad:** evaluaciones $=N+G\cdot N$. *Iteraciones* (generaciones) ≠ *evaluaciones*; el presupuesto compartido se cuenta en evaluaciones.

**Distancia de crowding.** Para cada objetivo $k$, se ordena el frente; los extremos reciben $\infty$ y cada interior suma
$\dfrac{f_k(i+1)-f_k(i-1)}{f_k^{\max}-f_k^{\min}}$. Se prefiere mayor distancia (zona menos poblada).

**Ventajas:** simple, elitista, sin parámetro de compartición, buen esparcimiento. **Limitaciones:** el crowding pierde eficacia con muchos objetivos
(≥4); el rendimiento depende de los operadores y de la codificación; con presupuestos pequeños puede no superar a una búsqueda aleatoria.
"""))

    c.append(code(r'''# Ejemplo numérico de NSGA-II: reemplazo elitista con 2N = 8 candidatos y N = 4 plazas.
import numpy as np
from src.pareto import ordenamiento_no_dominado, distancia_crowding, seleccionar_supervivientes

# columnas: (error de validación, parámetros). Primeras 4 = padres; últimas 4 = hijos.
F = np.array([[0.10, 300], [0.05, 900], [0.03, 2000], [0.20, 200],
              [0.06, 700], [0.04, 1200], [0.12, 350], [0.03, 2500]], float)
frentes = ordenamiento_no_dominado(F)
print("Frentes (índices):", frentes)
elegidos, rango, crowd = seleccionar_supervivientes(F, 4)
for i in range(len(F)):
    print(f"  ind {i}: err={F[i,0]:.2f} params={F[i,1]:>5.0f} rango={rango[i]} crowding={crowd[i]:.2f} "
          f"-> {'SOBREVIVE' if i in set(elegidos) else 'descartado'}")
print("El frente 0 completo cabe en las 4 plazas: se conservan compromisos distintos (barato/impreciso ... caro/preciso).")'''))

    c.append(md(r"""### 9.5 Búsqueda aleatoria
Consiste en muestrear configuraciones al azar del mismo espacio, evaluarlas y quedarse con las mejores (aquí, con el frente no dominado de lo evaluado).
Bergstra y Bengio (2012) la proponen como **línea base natural** para juzgar métodos adaptativos: es simple, no tiene sesgo de diseño y su rendimiento depende solo del presupuesto.
**Comparación justa:** mismo espacio de búsqueda, **mismo número de arquitecturas entrenadas**, mismo procedimiento de entrenamiento, mismas particiones, mismas semillas, y mismo
criterio de construcción del frente (aquí, el *archivo* de todo lo evaluado para ambos métodos).
"""))

    c.append(md(r"""## 10. Estado del arte y matriz bibliográfica
Los datos bibliográficos de la tabla fueron **comprobados en las páginas de la editorial, de arXiv o de índices bibliográficos (dblp) el 15-nov-2026**.
El estado de publicación se indica explícitamente: un *preprint* no se presenta como artículo arbitrado.

| ID | Referencia y enlace | Año / estado | Problema y método | Datos y comparación | Hallazgo y limitación | Relación con nuestra pregunta |
|---|---|---|---|---|---|---|
| F1 | Deb, Pratap, Agarwal, Meyarivan. *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.* IEEE TEVC 6(2):182–197. DOI [10.1109/4235.996017](https://doi.org/10.1109/4235.996017) | 2002 · artículo arbitrado (clásico) | Corrige costo, elitismo y compartición de NSGA; ordenamiento rápido + crowding | Problemas de prueba; compara con PAES y SPEA | Mejor esparcimiento y convergencia en la mayoría de problemas de prueba; evaluado en funciones sintéticas, no en entrenamiento de redes | **Método principal** que implementamos |
| F2 | Zitzler, Thiele. *Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach.* IEEE TEVC 3(4):257–271. DOI [10.1109/4235.797969](https://doi.org/10.1109/4235.797969) | 1999 · artículo arbitrado (clásico) | Compara MOEAs cuantitativamente y propone SPEA; formaliza la comparación de conjuntos no dominados | Problema de la mochila 0/1 extendido | Compara cuantitativamente cuatro MOEAs y propone SPEA; su caso es una mochila, no redes | Base de las métricas de comparación de conjuntos no dominados que usamos (espacio cubierto/hipervolumen y cobertura C) |
| F3 | Bergstra, Bengio. *Random Search for Hyper-Parameter Optimization.* JMLR 13(10):281–305. [jmlr.org](https://jmlr.org/papers/v13/bergstra12a.html) | 2012 · artículo arbitrado | Búsqueda aleatoria vs rejilla/manual para configurar redes | Redes y DBN en varios conjuntos | Aleatoria como línea base natural; la mayoría de hiperparámetros importa poco; es monoobjetivo | Justifica la **referencia** y su configuración |
| F4 | López-González, Rodríguez, Greiner. *A Multi-Objective Evolutionary Computation Approach for Improving Neural Network-Based Surrogate Models in Structural Engineering.* Algorithms 18(12):754. DOI [10.3390/a18120754](https://doi.org/10.3390/a18120754) | 2025 · artículo arbitrado (publicado 28-nov-2025) | NAS multiobjetivo con NSGA-II (precisión vs tiempo de entrenamiento) en sustitutos de pórticos de acero | Compara configuraciones de NSGA-II con búsqueda aleatoria; significancia sobre el hipervolumen (según su resumen) | Muestra utilidad de NSGA-II en NAS; dominio y objetivos distintos a los nuestros (tiempo, no parámetros; regresión) | **Antecedente más cercano**: mismo par de métodos y misma métrica |
| F5 | Vo, Luong. *Efficient Multi-Objective Neural Architecture Search via Pareto Dominance-based Novelty Search.* GECCO 2024. arXiv [2407.20656](https://arxiv.org/abs/2407.20656), DOI de actas [10.1145/3638529.3654064](https://doi.org/10.1145/3638529.3654064) | 2024 · aceptado como artículo completo en GECCO 2024 (según arXiv) | NAS multiobjetivo con métricas sin entrenamiento y novedad basada en dominancia | No verificado en detalle (solo se leyó el resumen); compara contra NAS multiobjetivo convencional según el resumen | Señala la convergencia prematura y pérdida de diversidad de métodos guiados por objetivos; usa métricas proxy, no entrenamiento | Contexto de **diversidad** y de los riesgos de convergencia en NAS multiobjetivo |
| F6 | Khaled, Kaymak, Genga. *Alternating Bi-Objective Optimization for Explainable Neuro-Fuzzy Systems.* arXiv [2602.19253](https://arxiv.org/abs/2602.19253) | 2026 · **preprint** (el documento del docente indica IEEE CAI 2026; no lo confirmé en la ficha de arXiv) | Compromiso exactitud–explicabilidad en ANFIS con optimización bi-objetivo por gradiente | ≈5000 experimentos en 9 conjuntos de regresión de UCI | Recupera soluciones fuera del casco convexo del frente evolutivo; objetivo de explicabilidad ≠ parámetros; sistemas neurodifusos | Muestra que **el compromiso se estudia con frentes** y que los métodos evolutivos son costosos |
| F7 | ACM SIGEVO. *GECCO 2026 — Neuroevolution track.* [gecco-2026.sigevo.org](https://gecco-2026.sigevo.org/Track?itemId=59) | 2026 · descripción oficial (contexto, no artículo) | Alcance del track: arquitecturas, métodos multiobjetivo, eficiencia | — | Delimita la línea de investigación vigente | Contextualiza la actualidad del tema |

**Nivel de lectura (honestidad bibliográfica).** F1, F3 y F4: resumen y partes del texto completo consultadas en línea; F2, F5, F6, F7: resumen e introducción. No se reprodujeron resultados publicados ni se compararon números con los nuestros, que usan otros datos y objetivos.

**Síntesis crítica.** Los trabajos coinciden en tres ideas: (1) el compromiso rendimiento–costo se estudia con **frentes de Pareto** y no con un único número (F1, F4, F6);
(2) la **búsqueda aleatoria** es una referencia razonable y exigente (F3, F4); y (3) los métodos guiados por objetivos pueden perder diversidad o converger prematuramente (F5). Difieren en el **objetivo de costo**
(tiempo de entrenamiento en F4, explicabilidad en F6, métricas proxy en F5, parámetros en nuestro caso), en el tipo de modelo y en la evidencia (F1 y F2 usan problemas sintéticos).
Por eso no podemos generalizar a nuestros datos las afirmaciones de superioridad de NSGA-II: **la pregunta pequeña** que sí podemos examinar es si, en un espacio MLP acotado, con 200 entrenamientos y
sobre Digits, NSGA-II mejora el hipervolumen frente a la búsqueda aleatoria.
"""))

    c.append(md(r"""## 11. Diseño metodológico

| Elemento | Decisión | Justificación |
|---|---|---|
| Método principal | NSGA-II implementado desde cero | Permite explicar y verificar cada operador; evita una dependencia opaca |
| Referencia | Búsqueda aleatoria (mismo espacio, muestreador, presupuesto) | Línea base estándar (Bergstra y Bengio, 2012) |
| Objetivo 1 | $f_1$ = error de clasificación en **validación** $=1-\text{accuracy}_{val}$ | Interpretable; Digits está balanceado (~10 % por clase), por lo que accuracy es razonable |
| Objetivo 2 | $f_2$ = número de parámetros entrenables $P$ (fórmula verificada) | Medida estructural exacta y determinista |
| Presupuesto | **200 evaluaciones reales** por método y corrida (= 200 redes entrenadas) | Se cuentan entrenamientos, no iteraciones; el evaluador lo impone y lo registra |
| NSGA-II | $N=20$, $G=9$ generaciones, cruce uniforme ($p=0{,}9$), mutación por gen ($p=1/5$), torneo binario por crowding | $20+9\cdot20=200$ |
| Repeticiones | **10 semillas** (0–9), diseño pareado | Mínimo sugerido por el docente |
| Particiones | Estratificadas 60/20/20, **una por semilla**, idénticas para ambos métodos | Controla la variabilidad de datos; prueba solo al final |
| Entrenamiento | `MLPClassifier`, Adam, lr = 0,01, 100 épocas, semilla de pesos = semilla de la corrida | Idéntico para toda arquitectura y método |
| Duplicados | Ningún método reevalúa una arquitectura ya evaluada | El entrenamiento es determinista: repetir desperdicia presupuesto |
| Pareado | La población inicial de NSGA-II = las primeras 20 muestras de la aleatoria | Reduce varianza; las diferencias nacen de cómo se gastan las 180 restantes |
| Métrica principal | Hipervolumen normalizado del **archivo** de arquitecturas evaluadas, referencia común $(0{,}9;\ P_{\max})$ | Resume convergencia y diversidad en un número; ver sección 19 |
| Métricas secundarias | Tamaño del frente, mejor error, parámetros mínimos con error ≤ mejor + 1 pp, tiempo, error de prueba de las soluciones elegidas | Complementan; el tiempo es solo informativo |
| Prueba estadística | Wilcoxon de rangos con signo (exacto, bilateral) sobre los 10 pares | No exige normalidad; tamaño de muestra pequeño |

**Referencias de contexto (fuera del presupuesto):** tres MLP manuales (16; 64; 64-64-64 neuronas, ReLU), entrenadas con el mismo procedimiento, para situar los resultados.

**Riesgo principal y alternativa:** que NSGA-II no supere a la aleatoria con 200 evaluaciones. Se reportaría igualmente (resultado negativo válido); la alternativa de menor alcance es el análisis del frente de una sola corrida.
"""))

    c.append(md(r"""## 12. Descripción del conjunto de datos
* **Nombre:** *Optical Recognition of Handwritten Digits* (copia del *test set* incluida en scikit-learn: `sklearn.datasets.load_digits`).
* **Origen:** E. Alpaydin y C. Kaynak (1998), Universidad de Boğaziçi; UCI Machine Learning Repository, DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49).
* **Licencia:** CC BY 4.0 (ficha de UCI). Se reconoce la autoría.
* **Contenido:** 1797 imágenes de 8×8 píxeles (enteros 0–16) → **64 variables**, 10 clases (dígitos 0–9), sin valores faltantes.
  La ficha de UCI describe 5620 instancias en total (3823 de entrenamiento y 1797 de prueba, de 43 personas); scikit-learn incluye únicamente las 1797 de prueba. Aquí **se re-particiona** ese conjunto 60/20/20 de forma estratificada.
* **Por qué Digits y no Breast Cancer.** En un piloto con 40 arquitecturas aleatorias (una partición, una semilla), Breast Cancer dio errores de validación en un rango estrecho (2,6 %–6,1 %; con ~114 muestras de validación, cada muestra equivale a ~0,9 puntos) y un frente de solo 4 arquitecturas distintas: casi todas las redes rinden parecido y el compromiso es poco visible.
  Digits dio errores de 1,9 % a 52,9 % y un frente de 8 arquitecturas distintas, de modo que el compromiso es observable. *(Es una decisión de diseño basada en un piloto, no un resultado de la investigación.)*
* **Limitación:** Digits es un problema fácil y de baja resolución; las conclusiones no se generalizan a datos más complejos.
"""))
    return c
