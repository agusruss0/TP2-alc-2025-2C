import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # sólo para activar el proyector 3D

from template_funciones import calculaLU, calcular_inversa

warnings.filterwarnings("ignore")
# Matriz A de ejemplo
# A_ejemplo = np.array([
#    [0, 1, 1, 1, 0, 0, 0, 0],
#    [1, 0, 1, 1, 0, 0, 0, 0],
#    [1, 1, 0, 1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1, 0, 0, 0],
#    [0, 0, 0, 1, 0, 1, 1, 1],
#    [0, 0, 1, 0, 1, 0, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 1],
#    [0, 0, 0, 0, 1, 1, 1, 0]
# ])


def calcula_L(A: np.ndarray) -> np.ndarray:
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    # Have fun!!
    rows, cols = A.shape
    K = np.zeros((rows, cols))  # Construyo una matriz de 0s con las dimensiones de A

    for i in range(rows):
        k_i = np.sum(A[i])  # sumo la fila i de A para obterner el grado del nodo i
        K[i, i] = (
            k_i  # Guardo el grado en la posicion ii. Asi obtenemos la matriz diagonal K con los grados de cada nodo
        )

    L = K - A  # Calculamos el Laplaciano
    return L


def calcula_R(A: np.ndarray) -> np.ndarray:
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    # Have fun!!
    # Calculamos el Laplaciano
    K = calcula_L(A) + A
    suma_total = np.sum(A)  # suma_total = 2*E
    rows, cols = A.shape
    P = np.zeros((rows, cols))  # Construyo una matriz de 0s con las dimensiones de A
    for i in range(rows):
        for j in range(cols):
            if i != j:
                P[i, j] = (
                    K[i, i] * K[j, j]
                ) / suma_total  # Probabilidad de que exista una arista entre i y j
    R = A - P  # Calculamos la matriz de modularidad
    return R


def calcula_lambda(L: np.ndarray, v: np.ndarray) -> float:
    # Recibe L y v y retorna el corte asociado
    # Have fun!
    s = np.sign(v)
    lambdon = np.multiply(1 / 4, s.T @ L @ s)
    return lambdon


def calcula_Q(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s = np.sign(v)
    Q = s.T @ R @ s
    # suma_total = (1 / (4 * E))
    return Q


def calcula_2E(A: np.ndarray) -> float:
    return np.sum(A)


def metpot1(
    A: np.ndarray,
    tol: float = 1e-8,
    maxrep: float = np.inf,
    plot: bool = False,
    seed: int = 100,
) -> tuple[np.ndarray, float, bool]:
    # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
    avec_aprox = []
    rows, cols = A.shape

    np.random.seed(seed)  # Fijamos la semilla para reproducibilidad
    vec = np.random.uniform(
        -1, 1, size=cols
    )  # Generamos un vector de partida aleatorio, entre -1 y 1
    vec = vec / np.linalg.norm(vec, 2)  # Lo normalizamos
    avec1 = A @ vec  # Aplicamos la matriz una vez
    avec1 = avec1 / np.linalg.norm(avec1, 2)  # normalizamos
    avec_aprox.append(avec1.copy())
    aval = (vec.T @ A @ vec) / (vec.T @ vec)  # Calculamos el autovalor estimado
    aval1 = (avec1.T @ A @ avec1) / (
        avec1.T @ avec1
    )  # Y el estimado en el siguiente paso
    nrep = 0  # Contador

    while (
        np.abs(aval1 - aval) / np.abs(aval) > tol and nrep < maxrep
    ):  # Si estamos por debajo de la tolerancia buscada
        vec = avec1  # actualizamos v y repetimos
        aval = aval1
        avec1 = A @ vec  # Calculo nuevo v1
        avec1 = avec1 / np.linalg.norm(avec1, 2)  # Normalizo
        avec_aprox.append(avec1.copy())
        aval1 = (avec1.T @ A @ avec1) / (avec1.T @ avec1)  # Calculo autovector
        nrep += 1  # Un pasito mas

    if not nrep < maxrep:
        print("MaxRep alcanzado")

    # Calculamos el autovalor
    avecs_aprox = np.vstack(avec_aprox)

    if plot:
        plot_avec_aprox(avecs_aprox)

    return avec1, aval1, nrep < maxrep


def deflaciona(A: np.ndarray, seed: int = 23) -> np.ndarray:
    # Recibe la matriz A, el autovector a remover y su autovalor asociado
    avec, aval, _ = metpot1(
        A, 1e-17, seed=seed
    )  # Calcula el primer autovector y autovalor de mayor modulo

    avec = avec / np.linalg.norm(avec, 2)  # Lo normalizamos
    deflA = A - aval * np.outer(
        avec, avec
    )  # Aplicamos la deflacion de Hotelling y obtenemos A deflacionada
    # Sugerencia, usar la funcion outer de numpy
    return deflA


def metpot2(
    A: np.ndarray, tol: float = 1e-17, maxrep: float = np.inf, seed: int = 23
) -> tuple[np.ndarray, float, bool]:
    # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
    # v1 y l1 son los primeors autovectores y autovalores de A}
    # Have fun
    deflA = deflaciona(A, seed)
    return metpot1(deflA, tol, maxrep, seed=seed)


def metpotI(
    A: np.ndarray, mu: float, tol: float = 1e-17, maxrep: float = np.inf, seed: int = 23
) -> tuple[np.ndarray, float, bool]:
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    row, cols = A.shape

    M = A + mu * np.eye(row, cols)
    M_inv = calcular_inversa(*calculaLU(M))

    return metpot1(M_inv, tol, maxrep, seed=seed)


def metpotI2(
    A: np.ndarray, mu: float, tol: float = 1e-17, maxrep: float = np.inf, seed: int = 23
) -> tuple[np.ndarray, float, bool]:
    # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A,
    # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
    # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.

    M = A + mu * np.eye(A.shape[0])  # Calculamos la matriz A shifteada en mu
    M_inv = calcular_inversa(*calculaLU(M))  # La invertimos

    v, l, _ = metpot2(M_inv, tol, maxrep, seed=seed)

    l = 1 / l  # Reobtenemos el autovalor correcto
    l -= mu
    return v, l, _


def laplaciano_iterativo(
    A: np.ndarray, niveles: int, nombres_s: list[str] = None, seed: int = 23
) -> list[list[str]]:
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if (
        nombres_s is None
    ):  # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if (
        A.shape[0] == 1 or niveles == 0
    ):  # Si llegamos al último paso, retornamos los nombres en una lista
        return [nombres_s]
    else:  # Sino:
        L = calcula_L(A)  # Recalculamos el L
        v, l, _ = metpotI2(
            L, 10e-4, seed=seed
        )  # ...  # Encontramos el segundo autovector de autovalor mas chico de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        s = np.sign(v)

        pos_i = [i for i in range(len(s)) if s[i] >= 0]
        neg_i = [i for i in range(len(s)) if s[i] < 0]

        Ap = A[pos_i][:, pos_i]  # Asociado al signo positivo
        Am = A[neg_i][:, neg_i]  # Asociado al signo negativo

        nombres_pos = [nombres_s[i] for i in pos_i]
        nombres_neg = [nombres_s[i] for i in neg_i]

        return laplaciano_iterativo(
            Ap, niveles - 1, nombres_s=nombres_pos, seed=seed
        ) + laplaciano_iterativo(Am, niveles - 1, nombres_s=nombres_neg, seed=seed)


def modularidad_iterativo(
    A: np.ndarray = None,
    R: np.ndarray = None,
    nombres_s: list[str] = None,
    seed: int = 23,
) -> list[list[str]]:
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print("Dame una matriz")
        return np.nan
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1:  # Si llegamos al último nivel
        return [nombres_s]  # ...
    else:
        v, l, _ = metpot1(R, seed=seed)  # ...  # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v > 0, :][:, v > 0]) + np.sum(R[v < 0, :][:, v < 0])  #
        if (
            Q0 <= 0 or all(v > 0) or all(v < 0)
        ):  # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return [nombres_s]
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            pos_i = [i for i in range(len(v)) if v[i] >= 0]
            neg_i = [i for i in range(len(v)) if v[i] < 0]
            Rp = R[pos_i][
                :, pos_i
            ]  # ...  # Parte de R asociada a los valores positivos de v
            Rm = R[neg_i][
                :, neg_i
            ]  # ...  # Parte asociada a los valores negativos de v
            vp, lp, _ = metpot1(Rp, seed=seed)  # ...  # autovector principal de Rp
            vm, lm, _ = metpot1(Rm, seed=seed)  # ...  # autovector principal de Rm

            nombres_pos = [nombres_s[i] for i in pos_i]
            nombres_neg = [nombres_s[i] for i in neg_i]
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp > 0) or all(vp < 0):
                Q1 = np.sum(Rp[vp > 0, :][:, vp > 0]) + np.sum(Rp[vp < 0, :][:, vp < 0])
            if not all(vm > 0) or all(vm < 0):
                Q1 += np.sum(Rm[vm > 0, :][:, vm > 0]) + np.sum(
                    Rm[vm < 0, :][:, vm < 0]
                )
            if (
                Q0 >= Q1
            ):  # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return [
                    nombres_pos,
                    nombres_neg,
                ]
            else:
                return modularidad_iterativo(
                    R, Rp, nombres_s=nombres_pos, seed=seed
                ) + modularidad_iterativo(R, Rm, nombres_s=nombres_neg, seed=seed)


def plot_avec_aprox(vecs: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: Convergencia de componentes
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.arange(len(vecs)), vecs[:, 0], label="Componente x")
    ax1.plot(np.arange(len(vecs)), vecs[:, 1], label="Componente y")
    ax1.plot(np.arange(len(vecs)), vecs[:, 2], label="Componente z")

    ax1.set_xlabel("Iteración")
    ax1.set_ylabel("Valor de la componente")
    ax1.set_title("Convergencia de cada componente")
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: Trayectoria en 3D
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(vecs[:, 0], vecs[:, 1], vecs[:, 2], marker="o", linewidth=1)
    ax2.scatter(
        vecs[-1][0],
        vecs[-1][1],
        vecs[-1][2],
        c="red",
        s=60,
        label="Autovector convergido",
    )

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("Trayectoria en R³")
    ax2.legend()

    plt.tight_layout()
    plt.show()
