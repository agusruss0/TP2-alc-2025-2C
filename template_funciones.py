import numpy as np
import scipy
import scipy.linalg


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
            C[i, k] = (1 / sumatoria) * F[i, k]

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
