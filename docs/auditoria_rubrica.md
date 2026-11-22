# Auditoría con la rúbrica del docente (100 puntos)

Autoevaluación del estado del proyecto. **No es una nota**: el docente asigna el valor según la evidencia. Aquí se indica qué evidencia existe y qué falta.

| Criterio (puntos) | Evidencia existente | Pendiente / riesgo |
|---|---|---|
| Pregunta, alcance y conexión con computación blanda (10) | Secciones 4–8 del cuaderno: pregunta medible, H1 operacionalizada, objetivos; componente blando = MLP + computación evolutiva (NSGA-II) | Ninguno técnico |
| Revisión crítica y actualidad (15) | Matriz de 7 fuentes (sección 10); 3 de 2024–2026 (López-González 2025, Vo y Luong 2024, Khaled 2026 preprint); síntesis crítica; datos verificados en línea | De varias fuentes solo se leyeron resumen e introducción (declarado). **Leer completos F4 y F5** antes de la defensa; cotejar nombres completos de autores (F4–F6); la sede IEEE CAI 2026 de Khaled no se confirmó |
| Diseño experimental y comparación justa (20) | Mismo espacio, muestreador, entrenamiento, particiones, semillas; presupuesto de 200 evaluaciones impuesto por código; diseño pareado; sin duplicados; prueba solo al final; referencia común de hipervolumen; Wilcoxon pareado | Población inicial compartida (declarado como decisión); una semilla controla partición, pesos y algoritmo (limitación 7) |
| Implementación y reproducibilidad (15) | Código modular; 60 pruebas pasan; cuaderno ejecutado desde kernel limpio sin errores; `requirements.txt` con versiones; `README.md`; CSV crudos; reproducción exacta verificada; ejecución reanudable | **No probado en Google Colab ni en otro sistema**; versiones exactas ancladas |
| Análisis, discusión y conclusiones (15) | Media, d.e., mediana e IQR; sensibilidad a población y presupuesto; robustez a la métrica; discusión con costo, fallos y condiciones de cambio; conclusiones acotadas; resultado no significativo reportado como tal | Potencia baja (10 pares); pruebas exploratorias sin corrección |
| Exposición y demostración de la pareja (10) | Presentación de 5 diapositivas con notas; guion para dos con tiempos; demo de ≈ 15 s y respaldo con resultados guardados | **No se ensayó**: los tiempos son estimaciones; cronometrar |
| Defensa conceptual individual (10) | Banco de ≈ 30 preguntas con respuestas fundamentadas | Depende de que **cada integrante estudie y pueda explicar** el código y los conceptos sin depender del otro |
| Contribución individual documentada (5) | Tabla de bitácora y declaración de IA preparadas | **Vacías a propósito**: deben completarse con datos reales de cada integrante (no se inventan) |

## Lista de entrega del docente
- [x] Cuaderno `.ipynb` con título definitivo. **Falta:** identificación de ambos integrantes (portada) y renombrar a `Investigacion_Tema05_Apellido1_Apellido2.ipynb` con sus apellidos reales.
- [x] Pregunta, hipótesis, alcance y conexión con computación blanda.
- [x] Matriz de fuentes (≥ 5; ≥ 2 de 2024–2026).
- [x] Método principal y referencia explicados.
- [x] Datos: origen, licencia (CC BY 4.0) y carga con `load_digits` (sin descarga externa).
- [x] Dependencias y versiones; semillas, particiones y presupuesto.
- [x] Resultados crudos (`resultados/corridas/`, `mediciones.csv` en el esquema del docente) y resumen con dispersión.
- [x] Figuras informativas (8) y análisis de sensibilidad acotado.
- [x] Discusión, conclusiones, límites y trabajo futuro.
- [ ] Bitácora y declaración de IA: plantillas listas, **completar con datos reales**.
- [x] Kernel reiniciado y celdas ejecutadas en orden (hecho en el entorno de desarrollo; **repetirlo en su equipo**).
- [ ] Demostración breve ensayada y copia de respaldo (los archivos existen; falta el ensayo).

## Observaciones de honestidad
- El resultado principal **no** confirma la hipótesis (p = 0,106); el informe y las diapositivas lo dicen así.
- No se midió energía ni latencia; el número de parámetros no se presenta como tal.
- Los tiempos de búsqueda se midieron con la máquina en distintos estados de carga (hubo reinicios del entorno); son informativos.
