"""Herramientas de Pareto para problemas de MINIMIZACIÓN con dos (o más) objetivos."""
import numpy as np


def _como_matriz(F):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F debe ser una matriz (n_soluciones, n_objetivos)")
    if not np.all(np.isfinite(F)):
        raise ValueError("F contiene valores no finitos")
    return F


def domina(a, b):
    """a domina a b (minimización): a <= b en todo y a < b en al menos un objetivo."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(a <= b) and np.any(a < b))


def ordenamiento_no_dominado(F):
    """Fast non-dominated sort (Deb et al., 2002). Devuelve lista de frentes (índices).

    Complejidad O(M N^2): para cada p cuenta cuántas soluciones lo dominan (n_p) y
    guarda las que él domina (S_p). El frente 0 son las de n_p = 0; al retirarlas se
    decrementa n_q de las dominadas y emerge el siguiente frente.
    """
    F = _como_matriz(F)
    n = len(F)
    dominadas_por = [[] for _ in range(n)]   # S_p
    n_dominan = np.zeros(n, dtype=int)       # n_p
    for p in range(n):
        for q in range(p + 1, n):
            if domina(F[p], F[q]):
                dominadas_por[p].append(q)
                n_dominan[q] += 1
            elif domina(F[q], F[p]):
                dominadas_por[q].append(p)
                n_dominan[p] += 1
    frentes = [[i for i in range(n) if n_dominan[i] == 0]]
    while frentes[-1]:
        siguiente = []
        for p in frentes[-1]:
            for q in dominadas_por[p]:
                n_dominan[q] -= 1
                if n_dominan[q] == 0:
                    siguiente.append(q)
        frentes.append(siguiente)
    frentes.pop()  # el último quedó vacío
    return frentes


def distancia_crowding(F_frente):
    """Distancia de crowding de las soluciones de UN frente (Deb et al., 2002).

    Para cada objetivo: se ordena el frente, los extremos reciben infinito y cada
    interior suma (f_{i+1} - f_{i-1}) / (f_max - f_min). Mayor distancia = región
    menos poblada = se prefiere para conservar diversidad.
    """
    F = _como_matriz(F_frente)
    n, m = F.shape
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist
    for j in range(m):
        orden = np.argsort(F[:, j], kind="stable")
        rango = F[orden[-1], j] - F[orden[0], j]
        dist[orden[0]] = dist[orden[-1]] = np.inf
        if rango == 0:
            continue
        for k in range(1, n - 1):
            dist[orden[k]] += (F[orden[k + 1], j] - F[orden[k - 1], j]) / rango
    return dist


def mascara_no_dominados(F):
    """Booleano por fila: True si la solución no es dominada por ninguna otra."""
    F = _como_matriz(F)
    mascara = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        for j in range(len(F)):
            if i != j and domina(F[j], F[i]):
                mascara[i] = False
                break
    return mascara


def hipervolumen_2d(F, referencia):
    """Hipervolumen exacto (minimización, 2 objetivos) respecto a `referencia`.

    Área dominada por el frente y acotada por el punto de referencia. Puntos que no
    mejoran estrictamente a la referencia en ambos objetivos no aportan área.
    """
    F = _como_matriz(F)
    if F.shape[1] != 2:
        raise ValueError("hipervolumen_2d requiere exactamente 2 objetivos")
    r = np.asarray(referencia, dtype=float)
    F = F[np.all(F < r, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[mascara_no_dominados(F)]
    F = F[np.argsort(F[:, 0], kind="stable")]  # f1 creciente => f2 decreciente
    hv, f2_previo = 0.0, r[1]
    for f1, f2 in F:
        if f2 < f2_previo:
            hv += (r[0] - f1) * (f2_previo - f2)
            f2_previo = f2
    return float(hv)


def cobertura(A, B):
    """Métrica C(A, B) de Zitzler: fracción de B dominada (o igualada) por alguna de A."""
    A, B = _como_matriz(A), _como_matriz(B)
    cubiertas = sum(any(domina(a, b) or np.array_equal(a, b) for a in A) for b in B)
    return cubiertas / len(B)


def seleccionar_supervivientes(F, n):
    """Reemplazo elitista: llena n plazas por frentes; el último se corta por crowding.

    Devuelve (índices_supervivientes, rango, crowding) donde rango/crowding están
    calculados sobre la población COMBINADA (se reutilizan para el torneo).
    """
    F = _como_matriz(F)
    if n > len(F):
        raise ValueError("No se pueden seleccionar más individuos que los disponibles")
    frentes = ordenamiento_no_dominado(F)
    rango = np.zeros(len(F), dtype=int)
    crowd = np.zeros(len(F))
    elegidos = []
    for r, frente in enumerate(frentes):
        rango[frente] = r
        crowd[frente] = distancia_crowding(F[frente])
        if len(elegidos) + len(frente) <= n:
            elegidos.extend(frente)
        elif len(elegidos) < n:
            faltan = n - len(elegidos)
            # mayor crowding primero; empate -> orden estable por índice
            orden = sorted(frente, key=lambda i: (-crowd[i], i))
            elegidos.extend(orden[:faltan])
    return np.array(elegidos), rango, crowd
