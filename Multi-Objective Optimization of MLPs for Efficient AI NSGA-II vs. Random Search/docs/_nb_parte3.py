"""Interpretaciones (escritas tras ver las salidas reales) y secciones 21-27."""

INTERPRETACIONES = {
"resumen": r"""**Resultado (10 corridas pareadas, 200 evaluaciones cada una).** El hipervolumen normalizado de NSGA-II tiene media 0,9158 (d.e. 0,0093) y mediana 0,9198 (IQR 0,0103); el de la búsqueda aleatoria, media 0,9121 (d.e. 0,0094) y mediana 0,9111 (IQR 0,0120).
El frente de NSGA-II tiene en promedio 13,5 arquitecturas y el de la aleatoria 11,9. El **mejor error de validación** es prácticamente el mismo (mediana 0,0181 frente a 0,0167; 5 de 10 pares empatan) y la arquitectura más pequeña que queda a ≤ 1 punto porcentual de ese mejor error tiene una mediana de 1206 parámetros en NSGA-II y 1285 en la aleatoria.
El tiempo de búsqueda (≈ 37 s frente a ≈ 39 s por corrida) es solo informativo: se midió con la máquina en estados de carga distintos (hubo reinicios del entorno) y **no es una medida de energía**.

**Integridad.** Las 40 corridas respetan el presupuesto exacto, sin duplicados ni valores no finitos. Solo 5 hijos de NSGA-II (de 5200 generados en las 30 corridas de NSGA-II) tuvieron que reemplazarse por una muestra aleatoria única, un efecto despreciable.
Entre el 49,5 % (aleatoria) y el 54,0 % (NSGA-II) de los entrenamientos **no convergieron** en 100 épocas: el error mide el rendimiento con un presupuesto de entrenamiento fijo, no el de redes entrenadas hasta converger (limitación, sección 22).""",

"fig123": r"""**Qué muestran.** Cada punto es una arquitectura entrenada; en gris están todas las evaluadas y en color las **no dominadas** (nadie las supera a la vez en error y en parámetros). El eje horizontal está en escala logarítmica.
**Resultado (semilla 0, ilustrativo, no evidencia por sí solo).** El frente tiene forma de "codo": el error cae de ≈ 0,36 con muy pocos parámetros a ≈ 0,06 hacia ≈ 550 parámetros, y luego mejora lentamente hasta ≈ 0,02 cerca de 1300. NSGA-II obtuvo 15 arquitecturas no dominadas y la aleatoria 12.
Como las primeras 20 evaluaciones son comunes, ambos frentes **coinciden en gran parte**; las diferencias visibles aparecen sobre todo entre ≈ 600 y 1300 parámetros, donde NSGA-II añade varias arquitecturas con errores de ≈ 0,03–0,04.
**Dominancia.** Todo punto gris está dominado por al menos un punto del frente: en promedio solo ≈ 7 % (NSGA-II) y ≈ 6 % (aleatoria) de las 200 evaluaciones llegan a ser no dominadas, es decir, ≈ 93–94 % son soluciones claramente dominadas.""",

"fig4": r"""**Qué muestra.** Las 2000 arquitecturas evaluadas por cada método (10 corridas), con error de validación frente a parámetros.
**Resultado.** En ambos métodos el error baja al aumentar los parámetros y se aplana hacia ≈ 1000–2000. Pero **a igual número de parámetros el error varía enormemente** (alrededor de 1000 parámetros hay desde ≈ 0,02 hasta ≈ 0,6): el número de parámetros no determina el desempeño; también importan la forma de la red (por ejemplo, cuellos de botella de pocas neuronas) y la activación.
La nube de NSGA-II es visiblemente más densa por debajo de ≈ 1000 parámetros. Descriptivamente, el **49,6 %** de sus evaluaciones usa menos de 1000 parámetros frente al **39,2 %** de la aleatoria (mediana de parámetros evaluados: 1020 frente a 1360).
**Interpretación (hipótesis, no probada):** NSGA-II reparte el presupuesto más hacia la zona de redes pequeñas, la que más pesa en el hipervolumen logarítmico.""",

"fig6": r"""**Qué muestran.** *Distribución:* hipervolumen por corrida; cada línea gris une los dos métodos con la misma semilla (misma partición de datos). *Convergencia:* hipervolumen medio (± 1 d.e.) del archivo según las evaluaciones consumidas.
**Resultado.** En 8 de 10 semillas NSGA-II supera a la aleatoria, pero con diferencias pequeñas (mediana de la diferencia +0,0024) frente a la variabilidad entre semillas (d.e. ≈ 0,009). Por eso el análisis **pareado** es esencial: la partición de datos influye más que el método.
En la convergencia, ambas curvas empiezan iguales (por construcción, 20 evaluaciones comunes), son prácticamente iguales a 40 evaluaciones, NSGA-II va por delante a 60, 140 y 200, y a 100 están prácticamente empatadas. Las bandas de ± 1 d.e. se solapan en todo el rango.
**Interpretación:** la ventaja de NSGA-II, si existe, es pequeña e intermitente en este experimento; la gráfica no sustenta una superioridad clara.""",

"stats": r"""**Resultado.** En ninguna de las seis comparaciones se rechaza $H_0$ con $\alpha=0{,}05$:
* *Hipervolumen (referencia 0,9, escala log):* NSGA-II gana en 8 de 10 pares, diferencia mediana +0,0024, $p=0{,}106$. Con referencia de error 0,5 el patrón es el mismo ($p=0{,}106$; son los mismos pares, no una evidencia independiente).
* *Hipervolumen con parámetros en escala lineal:* 5 victorias para cada método, diferencia mediana −0,0001, $p=0{,}49$. **La conclusión depende de cómo se valoran las redes pequeñas**: con escala lineal desaparece la ventaja.
* *Tamaño del frente:* 14 frente a 12 (medianas), 8–2, $p=0{,}18$. *Mejor error de validación:* sin diferencia ($p=0{,}56$, 5 empates). *Parámetros mínimos a ≤ +1 pp:* 1206 frente a 1285, $p=0{,}27$.
* *Cobertura de Zitzler (descriptiva):* en promedio, el 77,6 % del frente de la aleatoria queda dominado o igualado por el de NSGA-II, y el 62,4 % en sentido contrario; ambos valores son altos, en parte porque comparten las 20 primeras evaluaciones (explicación plausible, no verificada por separado).

**Interpretación.** Hay una **tendencia favorable a NSGA-II** en el hipervolumen logarítmico, pero con 10 pares la potencia es baja y no podemos afirmar ni superioridad ni equivalencia: un $p$ alto no demuestra que los métodos sean iguales. La hipótesis H1 **no queda confirmada** con este protocolo.""",

"test": r"""**Resultado.** Con las soluciones elegidas en validación y medidas una sola vez en prueba (mediana de 10 corridas):
* *Menor error de validación:* ≈ 2372 parámetros (NSGA-II) y ≈ 2410 (aleatoria); error de prueba 0,0292 en ambos ($p=0{,}88$).
* *Compacta (≤ +1 pp):* ≈ 1206 y ≈ 1285 parámetros; error de prueba 0,0375 en ambos ($p=0{,}31$).
* *Codo:* ≈ 498 y ≈ 460 parámetros; error de prueba 0,0736 (NSGA-II) y 0,0903 (aleatoria), pero con 5 empates, 3 victorias frente a 2 y $p=0{,}81$: no hay evidencia de diferencia, y los rangos intercuartílicos son amplios.
* *Optimismo de la validación:* el error de prueba supera al de validación en ≈ 1,1–1,9 puntos porcentuales (mediana de la brecha) en todos los criterios, como cabe esperar porque la validación se usó para guiar la búsqueda y para elegir.

**Qué cuesta reducir el tamaño.** Pasar de la solución de menor error (≈ 2400 parámetros, ≈ 2,9 % de error de prueba) a la compacta (≈ 1200–1300 parámetros, la mitad) cuesta ≈ 0,8 puntos porcentuales; llegar al codo (≈ 500 parámetros, −79 %) cuesta ≈ 4,4 puntos (NSGA-II) a ≈ 6,1 (aleatoria).

**Referencias manuales (mediana de 10 particiones; descriptivas, sin prueba estadística).** La red mayor del espacio (64-64-64, 13 130 parámetros) obtuvo 0,0306 de error de prueba: **no mejora** a la solución de menor error encontrada por la búsqueda (0,0292 con unas 5,5 veces menos parámetros), lo que es coherente con que no hace falta la red más grande.
Pero **la búsqueda tampoco superó con claridad a conjeturas razonables**: una red manual de 64 neuronas (4810 parámetros) llegó a 0,0236, y una de 16 neuronas (1210 parámetros, tamaño similar al de la solución compacta) a 0,0333, frente a 0,0375 de la compacta. Posible explicación (hipótesis): la elección sobre una validación pequeña (359 muestras) sobrestima la calidad de lo elegido.""",

"sens": r"""**Resultado — tamaño de población (presupuesto fijo 200).** La mediana del hipervolumen de NSGA-II es 0,9195 ($N=10$), 0,9198 ($N=20$) y 0,9181 ($N=50$), siempre por encima de la aleatoria (0,9111). Las diferencias pareadas son +0,0021 ($p=0{,}074$), +0,0024 ($p=0{,}106$) y +0,0017 ($p=0{,}0488$).
Solo $N=50$ queda por debajo de 0,05 y por muy poco; como se hicieron varias comparaciones exploratorias sin corregir, **no lo interpretamos como evidencia sólida**. Las medias casi no cambian con $N$ (0,9167; 0,9158; 0,9159): en este rango el tamaño de población no parece ser un factor decisivo.

**Resultado — presupuesto.** A 20 evaluaciones ambos son idénticos (por construcción). La diferencia mediana de hipervolumen es −0,0032 a 40, +0,0139 a 60 (8–2, $p=0{,}13$), +0,0003 a 100 (5–5), +0,0016 a 140 y +0,0024 a 200. Ninguna es significativa y el signo y la magnitud **no crecen de forma monótona** con el presupuesto.
**Interpretación:** la sensibilidad observada es pequeña frente a la variabilidad entre particiones; la ventaja de NSGA-II aparece como un patrón débil y no estable en este espacio y con estos presupuestos. Un presupuesto de 60 evaluaciones corresponde a solo 2 generaciones de NSGA-II, así que esa cifra no debe leerse como convergencia.""",
}


def parte3(md, code):
    c = []
    c.append(md(r"""## 21. Discusión
Separamos **resultados** (medidos), **interpretaciones** (nuestra lectura) e **hipótesis** (no comprobadas).

**¿NSGA-II encontró mejores compromisos que la búsqueda aleatoria?**
*Resultado:* el hipervolumen logarítmico fue mayor en 8 de 10 pares (+0,0024 de mediana, $p=0{,}106$), el frente algo más grande (13,5 frente a 11,9) y la cobertura mayor; con parámetros en escala lineal no hubo ventaja (5–5), y a lo largo del presupuesto la ventaja fue intermitente.
*Interpretación:* hay una tendencia favorable pero **no concluyente**; no corresponde afirmar superioridad. *Hipótesis:* la tendencia proviene de que NSGA-II invierte más evaluaciones en redes pequeñas (49,6 % bajo 1000 parámetros frente a 39,2 %); no se comprobó causalmente.

**¿Qué arquitecturas ofrecen menor error?** Las de ≈ 2400 parámetros (mediana) tienen el menor error de validación (≈ 1,8 %) y ≈ 2,9 % en prueba; ambos métodos encuentran errores equivalentes (5 empates en 10 pares, $p=0{,}56$).
**¿Cuáles tienen menos parámetros?** Los extremos del frente se acercan al mínimo del espacio (160 parámetros), pero con errores de validación muy altos (p. ej. ≈ 0,36 en la semilla 0).
**¿Qué sacrificio implica reducir el tamaño?** Ver sección 19: ≈ 0,8 puntos porcentuales al reducir a la mitad los parámetros, y > 4 puntos al reducirlos ≈ 79 %. El compromiso es real y **no lineal**: el primer recorte es barato; el segundo, caro.
**¿Hay soluciones claramente dominadas?** Sí: ≈ 93–94 % de las evaluaciones de cada corrida.

**¿Mejora NSGA-II lo suficiente para justificar su costo?** El costo de cómputo es el mismo por diseño (200 entrenamientos) y el tiempo de búsqueda fue similar (≈ 37 s frente a ≈ 39 s, dominado por el entrenamiento y con las salvedades de la sección 22). El costo adicional es de **implementación y explicación**: más código, cinco hiperparámetros y operadores que justificar. Con la evidencia obtenida (ventaja pequeña, no significativa y sensible a la escala) **no podemos afirmar que se justifique**; la búsqueda aleatoria resultó una referencia exigente, como anticipa Bergstra y Bengio (2012).

**¿Qué falló y qué aprendimos?** (1) Entre el 50 % y el 54 % de los entrenamientos no convergieron: el error depende del presupuesto de entrenamiento fijado. (2) La validación guía y elige a la vez, y la prueba final reveló un optimismo de ≈ 1–2 pp. (3) El análisis de robustez mostró que la ventaja depende de la escala del hipervolumen: la métrica debe fijarse y justificarse **antes** de ver resultados. (4) Una primera versión de la Figura 5 usaba barras sobre escala logarítmica, que engañan por depender de un origen arbitrario; se rehízo con puntos y barras de error. (5) La ejecución se interrumpió dos veces: se diseñó el protocolo para ser reanudable y determinista.

**¿En qué condiciones podría cambiar la conclusión?** Con más semillas la tendencia podría volverse significativa o diluirse; con parámetros en escala lineal o con otra referencia de hipervolumen no hay ventaja visible; con más épocas de entrenamiento, otro conjunto de datos, otro presupuesto u operadores mejor ajustados el orden entre métodos podría cambiar. No lo evaluamos.

**¿Cómo elegir para un dispositivo limitado?** El frente se usa como un menú: (1) fijar el presupuesto de memoria (p. ej., a 4 bytes por parámetro en coma flotante de 32 bits, 1250 parámetros ocupan ≈ 5 KB: es aritmética, no una medición); (2) tomar del frente de validación la solución más precisa que quepa;
(3) verificar una sola vez en prueba; (4) **medir** latencia y energía en el dispositivo real, algo que aquí no se hizo y que los parámetros no sustituyen.
**¿Qué cambiaría con otro conjunto de datos?** *Hipótesis:* en problemas más fáciles o saturados (como mostró nuestro piloto con Breast Cancer) el frente sería corto y NSGA-II aportaría poco; en problemas más difíciles el compromiso sería más marcado y las diferencias entre métodos podrían ampliarse o desaparecer. No lo evaluamos.

**Relación con la literatura.** López-González et al. (2025) también compararon NSGA-II con búsqueda aleatoria mediante hipervolumen, con otros objetivos y dominio; no comparamos cifras. Vo y Luong (2024) advierten de la pérdida de diversidad en NAS multiobjetivo; nuestro frente algo mayor es compatible con diversidad adecuada, pero no la medimos directamente."""))
    c.append(md(r"""## 22. Limitaciones
1. **Un solo conjunto de datos, fácil y de baja resolución** (Digits 8×8); solo MLP densas de ≤ 3 capas y ≤ 64 neuronas. No se generaliza a otras tareas o modelos.
2. **Muestra pequeña:** 10 pares. La potencia es baja; un $p$ alto no prueba equivalencia. Varias comparaciones exploratorias **sin corrección** por pruebas múltiples; el $p=0{,}0488$ ($N=50$) es marginal.
3. **El número de parámetros no es energía ni latencia.** No se midió consumo energético ni tiempo de inferencia; no afirmamos ahorro energético. El tiempo de búsqueda es aproximado (carga variable de la máquina, reinicios del entorno).
4. **Entrenamiento corto y ruidoso:** 100 épocas fijas; 49,5–54,0 % de los entrenamientos no convergieron. Una sola inicialización de pesos por corrida (la misma para todas las arquitecturas), sin promediar el ruido de entrenamiento; el orden de arquitecturas podría cambiar con más épocas.
5. **Validación pequeña y reutilizada** (359 muestras; cada muestra = 0,28 pp): sirve para guiar la búsqueda y para elegir; produce optimismo (brecha de ≈ 1–2 pp respecto a prueba) y errores discretos.
6. **La métrica principal depende de decisiones:** referencia (0,9; $P_{\max}$) y escala logarítmica de parámetros. Con escala lineal desaparece la ventaja de NSGA-II.
7. **Diseño pareado con población inicial compartida** y una única semilla que controla a la vez la partición, la inicialización y el algoritmo: no se separa la variabilidad de datos de la del algoritmo.
8. **NSGA-II sin ajuste fino:** operadores y tasas fijados a priori; solo se varió $N$; 9 generaciones son pocas. Un mejor ajuste o más presupuesto podría cambiar el resultado.
9. **Reproducibilidad entre entornos:** verificada en el mismo entorno (versiones en `resultados/entorno.json`); no se probó en Colab ni en otros sistemas, y diferencias de bibliotecas numéricas podrían alterar los decimales.
10. **Literatura:** de varias fuentes se leyeron solo el resumen y la introducción; no se reprodujeron sus resultados."""))
    c.append(md(r"""## 23. Conclusiones
Las conclusiones se limitan a lo que los experimentos respaldan:
1. **H1 no queda confirmada.** Con 200 evaluaciones y 10 pares, NSGA-II tuvo mayor hipervolumen logarítmico en 8 de 10 corridas (+0,0024 de mediana), pero $p=0{,}106$; no rechazamos $H_0$. Tampoco demostramos equivalencia.
2. **La ventaja no es robusta:** desaparece con parámetros en escala lineal, es intermitente según el presupuesto (−0,0032 a +0,0139) y, según el tamaño de población, solo $N=50$ cruzó $p<0{,}05$, por poco y sin corrección.
3. **Descriptivamente**, NSGA-II invierte más evaluaciones en redes pequeñas (49,6 % frente a 39,2 % bajo 1000 parámetros) y obtiene frentes algo mayores (13,5 frente a 11,9). Es una hipótesis explicativa compatible con los datos, no un hecho probado.
4. **El compromiso error–tamaño es real:** reducir a la mitad los parámetros (≈ 2400 → ≈ 1250) costó ≈ 0,8 pp de error de prueba; reducirlos ≈ 79 % costó > 4 pp. La red más grande (13 130 parámetros) no superó a la mejor solución encontrada (0,0306 frente a 0,0292), aunque la búsqueda tampoco superó con claridad a las redes manuales de 64 y 16 neuronas.
5. **Metodología:** ambos métodos usaron exactamente el mismo presupuesto, particiones y entrenamiento (verificado por código y pruebas); la prueba solo se usó al final.

**Lo que NO puede concluirse:** ni la superioridad ni la equivalencia de NSGA-II (muestra pequeña, p > 0,05); nada sobre consumo de energía, latencia o memoria real (solo se cuenta el número de parámetros); nada sobre otros conjuntos de datos, tipos de red o presupuestos; y nada causal sobre *por qué* NSGA-II tiende a explorar redes pequeñas (es una hipótesis).

## 24. Trabajo futuro
* Más semillas (≥ 30) y corrección por pruebas múltiples; separar la semilla de partición, de inicialización y del algoritmo.
* Más conjuntos de datos (incluyendo uno donde el compromiso sea más exigente) y más objetivos reales: latencia o FLOPs medidos y **energía medida** en un dispositivo.
* Entrenar hasta convergencia (parada temprana) y promediar varias inicializaciones por arquitectura; usar validación cruzada o un conjunto de selección separado.
* Líneas base adicionales (muestreo por hipercubo latino, evolución regularizada) y otros algoritmos (NSGA-III, MOEA/D); ajustar operadores y tasas.
* Explorar si la mayor densidad de NSGA-II en redes pequeñas explica la tendencia (hipótesis de la sección 21)."""))
    c.append(md(r"""## 25. Referencias
Datos bibliográficos comprobados en línea el 15-nov-2026 salvo donde se indica.
1. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation, 6*(2), 182–197. https://doi.org/10.1109/4235.996017
2. Zitzler, E., & Thiele, L. (1999). Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach. *IEEE Transactions on Evolutionary Computation, 3*(4), 257–271. https://doi.org/10.1109/4235.797969
3. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
4. López-González, Rodríguez y Greiner (2025). A multi-objective evolutionary computation approach for improving neural network-based surrogate models in structural engineering. *Algorithms, 18*(12), 754. https://doi.org/10.3390/a18120754 *(solo apellidos: cotejar nombres completos en la página de la editorial antes de entregar)*
5. Vo y Luong (2024). Efficient multi-objective neural architecture search via Pareto dominance-based novelty search. *Proceedings of GECCO 2024*. arXiv:2407.20656; https://doi.org/10.1145/3638529.3654064 *(aceptado como artículo completo según arXiv; cotejar nombres completos)*
6. Khaled, Kaymak y Genga (2026). Alternating bi-objective optimization for explainable neuro-fuzzy systems. *arXiv preprint* arXiv:2602.19253. https://arxiv.org/abs/2602.19253 *(preprint; no arbitrado hasta donde pudimos verificar; cotejar nombres completos)*
7. ACM SIGEVO. (2026). *GECCO 2026 — Neuroevolution track*. https://gecco-2026.sigevo.org/Track?itemId=59 *(descripción oficial del track; no es un artículo)*
8. Alpaydin, E., & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49 (licencia CC BY 4.0; ficha consultada en línea).
9. Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. *(cita del software; no se verificó en línea en esta sesión)*"""))
    c.append(md(r"""## 26. Declaración de uso de inteligencia artificial
Se utilizó un asistente de IA (Claude, de Anthropic) como apoyo para: delimitar el proyecto a partir del documento del docente, buscar y comprobar fuentes en línea, diseñar el experimento, escribir y probar el código, ejecutar los experimentos en su entorno, generar las figuras y redactar los textos del cuaderno.
**Qué se verificó:** el código se comprobó con 60 pruebas automáticas (ver sección 14), las cifras del texto provienen de las salidas de las celdas, y las referencias se contrastaron con las páginas de las editoriales, arXiv, dblp o UCI.
**Qué corresponde a los integrantes:** leer y comprender cada decisión, volver a ejecutar el cuaderno, cotejar las referencias marcadas y poder defender individualmente el trabajo. *(Completar: qué partes revisó, modificó o rehízo cada integrante.)*"""))
    c.append(md(r"""## 27. Bitácora y contribuciones de ambos integrantes
**Registro técnico del experimento (hechos verificables):**

| Hecho | Detalle |
|---|---|
| Piloto de selección de datos | 40 arquitecturas aleatorias en Digits y en Breast Cancer; se eligió Digits |
| Protocolo ejecutado | 10 semillas × 2 métodos × 200 evaluaciones (principal) + NSGA-II con $N=10$ y $N=50$ |
| Interrupciones | La ejecución se interrumpió dos veces por reinicio del entorno y se **reanudó** sin pérdida (cada corrida se guarda al terminar y es determinista) |
| Verificaciones | 60 pruebas automáticas; reproducción exacta de 60 evaluaciones; comprobaciones de integridad (sección 17) |

**Contribuciones de los integrantes** *(completar con datos reales; no se inventan):*

| Fecha | Actividad | Integrante 1 | Integrante 2 |
|---|---|---|---|
| *(…)* | *(…)* | *(…)* | *(…)* |"""))
    return c
