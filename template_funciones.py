import numpy as np
import scipy
import scipy.linalg
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import geopandas as gpd
import warnings

warnings.filterwarnings("ignore")

def construye_adyacencia(D: np.ndarray, m: int) -> np.ndarray:
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = []  # Lista para guardar las filas
    for fila in D:  # recorriendo las filas, anexamos vectores lógicos
        l.append(
            fila <= fila[np.argsort(fila)[m]]
        )  # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int)  # Convertimos a entero
    np.fill_diagonal(A, 0)  # Borramos diagonal para eliminar autolinks
    return A


def calculaLU(matriz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la descomposición LU de una matriz.

    Args:
        matriz (np.ndarray): La matriz a descomponer.

    Returns:
        tuple[np.ndarray, np.ndarray]: La descomposición LU de la matriz.
    """
    rows, cols = matriz.shape
    L = np.eye(rows, cols)
    U = matriz.copy()
    cant_op = 0

    if rows != cols:
        raise ValueError("La matriz no es cuadrada.")

    for i in range(rows):
        for j in range(i + 1, rows):
            factor = U[j, i] / U[i, i]  # Calculo del factor de eliminacion
            U[j, i:] = U[j, i:] - factor * U[i, i:]  # Esacalonamos las filas
            L[j, i] = factor  # Guardamos el factor de eliminacion en L

            cant_op += 2  # DIVISION Y ASIGNACION
            cant_op += cols - i  # RESTAS

    # print(f"Cantidad de operaciones: {cant_op}")

    return L, U


def calcula_matriz_C(A: np.ndarray) -> np.ndarray:
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    rows, cols = A.shape
    K_inv = np.zeros((rows, cols))

    for i in range(rows):
        k = np.sum(A[i])
        K_inv[i, i] = 1 / k

    C = A.transpose() @ K_inv  # Calcula C multiplicando Kinv y A
    return C


def calcula_pagerank(A: np.ndarray, alfa: float) -> np.ndarray:
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[
        0
    ]  # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = (N / alfa) * (np.eye(N) - (1 - alfa) * C)  # Matriz M de page rank
    L, U = calculaLU(M)  # Calculamos descomposición LU a partir de C y d
    b = np.ones(
        N
    )  # * alfa / N   # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N. TODO: Aca tengo dudas.
    Up = scipy.linalg.solve_triangular(
        L, b, lower=True
    )  # Primera inversión usando L. Resuelve L.x = b con x = Up
    p = scipy.linalg.solve_triangular(
        U, Up
    )  # Segunda inversión usando U. Resuelve U.y = Up
    return p


# TODO: Revisar.-
def calcula_matriz_C_continua(D: np.ndarray) -> np.ndarray:
    # Función para calcular la matriz de trancisiones C
    # D: Matriz de distancias
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1 / D
    C = np.zeros(D.shape)
    np.fill_diagonal(F, 0)

    for i in range(F.shape[0]):
        sumatoria = np.sum(F[i])
        for k in range(F.shape[1]):
            # Multiplicamos por la distancia inversa
            C[k,i] = (1 / sumatoria) * F[i, k]

    return C


def calcula_B(C: np.ndarray, cantidad_de_visitas: int) -> np.ndarray:
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v

    B = np.eye(C.shape[0])
    C_copy = C.copy()
    for i in range(0, cantidad_de_visitas):
        if i == 0:
            continue
        if i == 1:
            B += C_copy
        else:
            C_copy = C @ C_copy
            B += C_copy

    return B


def calcular_inversa(L: np.ndarray, U: np.ndarray) -> np.ndarray:
    n, m = L.shape
    Id = np.eye(n, m)
    Inv = np.zeros((n, m))
    try:
        for i in range(n):
            y = scipy.linalg.solve_triangular(L, Id[:, i], lower=True)
            x = scipy.linalg.solve_triangular(U, y)
            Inv[:, i] = x

        return Inv
    except:
        print("La matriz no es inversible")

def graficar_red(
    p_rank: np.ndarray,
    G: nx.Graph,
    G_layout: dict,
    Nprincipales: int,
    ax: Axes,
    color: bool = True,
) -> np.ndarray:
    """
    Grafica la red con el Page Rank.

    Args:
        p_rank (np.ndarray): Vector de PageRank.
        G (nx.Graph): Grafo de la red.
        G_layout (dict): Layout del grafo.
        Nprincipales (int): Número de nodos principales a graficar.
        ax (Axes): Axes del gráfico.
        color (bool): Si True, colorea los nodos principales de rojo.
    """
    factor_escala = 1e4  # Escalamos los nodos 10 mil veces para que sean bien visibles
    pr = p_rank  # np.random.uniform(0,1,museos.shape[0])# Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar
    pr = pr / pr.sum()  # Normalizamos para que sume 1
    principales = np.argsort(pr)[-Nprincipales:]  # Identificamos a los N principales
    colores = ["orange" if n in principales else "#8fbcd4" for n in G.nodes]
    labels = {
        n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)
    }  # Nombres para esos nodos
    if color:
        nx.draw_networkx(
            G,
            G_layout,
            node_size=pr * factor_escala,
            ax=ax,
            with_labels=False,
            node_color=colores,
        )  # Graficamos red
    else:
        nx.draw_networkx(
            G, G_layout, node_size=pr * factor_escala, ax=ax, with_labels=False
        )  # Graficamos red
    nx.draw_networkx_labels(
        G, G_layout, labels=labels, ax=ax, font_size=8, font_color="k"
    )  # Agregamos los nombres
    return principales

# TODO: Pasar a template_funciones.py
def graficar_red_periferia(
    p_rank: np.ndarray,
    G: nx.Graph,
    G_layout: dict,
    Nprincipales: int,
    ax: Axes,
    nodos_destacados: list[int],
    nodos_internos: list[int],
) -> (
    np.ndarray
):  # Graficamos la red con el Page Rank  #TODO: Pasar a template_funciones.py
    """
    Grafica los nodos de la red en el mapa.

    Args:
        p_rank (np.ndarray): Vector de PageRank.
        G (nx.Graph): Grafo de la red.
        G_layout (dict): Layout del grafo.
        Nprincipales (int): Número de nodos principales a graficar.
        ax (Axes): Axes del gráfico.
        nodos_destacados (list[int]): Lista de nodos destacados.
        nodos_internos (list[int]): Lista de nodos internos.
    """
    factor_escala = 1e4  # Escalamos los nodos 10 mil veces para que sean bien visibles
    pr = p_rank  # np.random.uniform(0,1,museos.shape[0])# Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar
    pr = pr / pr.sum()  # Normalizamos para que sume 1
    principales = np.argsort(pr)[-Nprincipales:]  # Identificamos a los N principales
    colores_nodos = []
    for n in G.nodes:
        if n in nodos_destacados:
            colores_nodos.append("red")
        elif n in nodos_internos:
            colores_nodos.append("green")
        else:
            colores_nodos.append("#8fbcd4")
    # colores_nodos = ['red' if n in nodos_destacados else '#8fbcd4' for n in G.nodes]
    labels = {
        n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)
    }  # Nombres para esos nodos
    nx.draw_networkx(
        G,
        G_layout,
        node_size=pr * factor_escala * 1.40,
        ax=ax,
        with_labels=False,
        node_color=colores_nodos,
    )  # Graficamos red
    #nx.draw_networkx_labels(
    #    G, G_layout, labels=labels, ax=ax, font_size=0, font_color="k"
    #)  # Agregamos los nombres
    return principales


def graficar_nodos(
    v: np.ndarray, G: nx.Graph, G_layout: dict, barrios: gpd.GeoDataFrame
) -> None:  # Graficamos la red con el Page Rank  #TODO: Pasar a template_funciones.py
    """
    Grafica los nodos de la red en el mapa.

    Args:
        v (np.ndarray): Vector de PageRank.
        G (nx.Graph): Grafo de la red.
        G_layout (dict): Layout del grafo.
        barrios (gpd.GeoDataFrame): DataFrame con los barrios.
    """
    factor_escala = 1e4  # Escalamos los nodos 10 mil veces para que sean bien visibles
    fig, ax = plt.subplots(figsize=(10, 10))  # Visualización de la red en el mapa
    barrios.to_crs("EPSG:22184").boundary.plot(
        color="gray", ax=ax
    )  # Graficamos Los barrios
    pr = v  # np.random.uniform(0,1,museos.shape[0])# Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar
    pr = pr / pr.sum()  # Normalizamos para que sume 1    # Alto
    plt.title("Cantidad de visitas iniciales por museo")
    nx.draw_networkx_nodes(
        G, G_layout, node_size=pr * factor_escala, ax=ax
    )  # Graficamos red

def resolver_sistema(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    L, U = calculaLU(A)

    x = scipy.linalg.solve_triangular(L, b, lower=True)
    v = scipy.linalg.solve_triangular(U, x)

    return v

def norma_1(v: np.ndarray) -> float:
    v_abs = np.abs(v)
    return np.sum(v_abs)