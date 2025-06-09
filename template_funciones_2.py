import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # sólo para activar el proyector 3D

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
    return R


def calcula_lambda(L: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Recibe L y v y retorna el corte asociado
    # Have fun!
    lambdon = np.multiply(1 / 4, v.T @ L @ v)
    return lambdon


def calcula_Q(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    return Q


def metpot1(
    A: np.ndarray, tol: float = 1e-8, maxrep: float = np.inf, plot: bool = False
) -> tuple[np.ndarray, float, bool]:
    # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
    avec_aprox = []
    rows, cols = A.shape

    vec = np.random.random(
        cols
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

    aval = (avec1.T @ A @ avec1) / (avec1.T @ avec1)  # Calculamos el autovalor
    avec_aprox = np.vstack(avec_aprox)

    if plot:
        plot_avec_aprox(avec_aprox)

    return avec1, aval, nrep < maxrep


def deflaciona(A: np.ndarray, tol: float = 1e-8, maxrep: float = np.inf) -> np.ndarray:
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1, l1, _ = metpot1(
        A, tol, maxrep
    )  # Buscamos primer autovector con método de la potencia
    deflA = (
        A.copy() - l1 * v1 @ v1.T / v1.T @ v1
    )  # Sugerencia, usar la funcion outer de numpy
    return deflA


def metpot2(
    A: np.ndarray, v1: np.ndarray, l1: float, tol: float = 1e-8, maxrep: float = np.inf
) -> tuple[np.ndarray, float, bool, np.ndarray]:
    # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
    # v1 y l1 son los primeors autovectores y autovalores de A}
    # Have fun!
    deflA = deflaciona(A)
    return metpot1(deflA, tol, maxrep)


def metpotI(
    A: np.ndarray, mu: float, tol: float = 1e-8, maxrep: float = np.inf
) -> tuple[np.ndarray, float, bool, np.ndarray]:
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    row, cols = A.shape

    A_mu = A - mu * np.eye(row, cols)
    A_mu_inv = np.linalg.inv(A_mu)
    return metpot1(A_mu_inv, tol=tol, maxrep=maxrep)


def metpotI2(
    A: np.ndarray, mu: float, tol: float = 1e-8, maxrep: float = np.inf
) -> tuple[np.ndarray, float, bool, np.ndarray]:
    # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A,
    # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
    # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
    X = ...  # Calculamos la matriz A shifteada en mu
    iX = ...  # La invertimos
    defliX = ...  # La deflacionamos
    v, l, _ = ...  # Buscamos su segundo autovector
    l = 1 / l  # Reobtenemos el autovalor correcto
    l -= mu
    return v, l, _


def laplaciano_iterativo(
    A: np.ndarray, niveles: int, nombres_s: list[str] = None
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
        v, l, _ = ...  # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = ...  # Asociado al signo positivo
        Am = ...  # Asociado al signo negativo

        return laplaciano_iterativo(
            Ap, niveles - 1, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi > 0]
        ) + laplaciano_iterativo(
            Am, niveles - 1, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi < 0]
        )


def modularidad_iterativo(
    A: np.ndarray = None, R: np.ndarray = None, nombres_s: list[str] = None
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
        return ...
    else:
        v, l, _ = ...  # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v > 0, :][:, v > 0]) + np.sum(R[v < 0, :][:, v < 0])
        if (
            Q0 <= 0 or all(v > 0) or all(v < 0)
        ):  # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return ...
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = ...  # Parte de R asociada a los valores positivos de v
            Rm = ...  # Parte asociada a los valores negativos de v
            vp, lp, _ = ...  # autovector principal de Rp
            vm, lm, _ = ...  # autovector principal de Rm

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
                    [ni for ni, vi in zip(nombres_s, v) if vi > 0],
                    [ni for ni, vi in zip(nombres_s, v) if vi < 0],
                ]
            else:
                # Sino, repetimos para los subniveles
                return ...


def plot_avec_aprox(vecs: np.ndarray) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(vecs)), vecs[:, 0], label="Componente x")

    plt.plot(np.arange(len(vecs)), vecs[:, 1], label="Componente y")

    plt.plot(np.arange(len(vecs)), vecs[:, 2], label="Componente z")

    plt.xlabel("Iteración")
    plt.ylabel("Valor de la componente")
    plt.title("Convergencia de cada componente del autovector")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(vecs[:, 0], vecs[:, 1], vecs[:, 2], marker="o", linewidth=1)

    ax.scatter(*vecs[-1], c="red", s=60, label="Autovector convergido")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trayectoria de las aproximaciones en R³")
    ax.legend()
    plt.tight_layout()
    plt.show()
