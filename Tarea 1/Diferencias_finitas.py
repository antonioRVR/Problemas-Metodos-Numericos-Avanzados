


def solve_finite_differences(N):
    from Matrix_solver import Solve_Gaussian_el, solve_thomas_algorithm
    import numpy as np
    """
    Resuelve el problema de contorno usando el método de diferencias finitas.
    """
    # 1. Definir el dominio y la malla de puntos
    x_start, x_end = -1.0, 1.0
    h = (x_end - x_start) / (N + 1) 

    # print(f"Paso de malla h: {h}")

    # Puntos interiores de la malla (donde m es desconocido)
    x_interior = np.linspace(x_start + h, x_end - h, N)
    # print(f"Puntos interiores x: {x_interior}")

    # 2. Construir la matriz A (tamaño (N) x (N))
    # Diagonal principal
    diag_main = h**2 * (1 + x_interior**2) - 2
    
    # Diagonales superior e inferior (todo unos)
    diag_upper = np.ones(N - 1)
    diag_lower = np.ones(N - 1)
    

    # Ensamblar la matriz tridiagonal
    A = np.diag(diag_main) + np.diag(diag_upper, k=1) + np.diag(diag_lower, k=-1)

    # print("Matriz A:")
    # print(A)

    # 3. Construir el vector b
    b = -h**2 * np.ones(N)

    # print("Vector b:")
    # print(b)
    

    #Problema Restricciones ejercicio: arreglar siguiente bloque
#  --------------------------------------------------------------- 
    # 4. Resolver el sistema lineal A * m = b
    # NOTA: En este punto se puede implementar cualquier método de resolución de sistemas lineales
    # (eliminación gaussiana, LU, Cholesky, Jacobi, Gauss-Seidel, SOR, etc.).
    # Sin embargo, para simplificar el código y centrarnos en el método de diferencias finitas,
    # vamos a utilizar la función numpy.linalg.solve() que internamente utiliza LAPACK.

    # se admite el uso de funciones auxiliares de ayuda. Por ejemplo, una función que 
    # resuelva un sistema de ecuaciones lineales. Pero en caso de hacer uso de ellas debe especificarse
    #  cuál es el método de resolución que se utiliza para obtener los valores. 
    # Así, si se utiliza la función numpy.linalg.inv() de python deberá indicarse que internamente se
    #  hace uso de la función dgetri de LAPACK, que a su vez calcula la inversa
    #  a partir de una descomposición LU. En muchas ocasiones será más sencillo 
    # implementar subrutinas propias que averiguar qué hacen muchas de las funciones
    #  desarrolladas externamente.


# A opcion simple con numpy
    # m_interior = np.linalg.solve(A, b)

# B opcion con eliminacion gaussiana. La convergencia es muy mala y tarda mucho. Buscamos soluciones mas eficientes
    m_interior = Solve_Gaussian_el(A.tolist(), b.tolist())

# C opcion con algoritmo de Thomas (específico para matrices tridiagonales)
    # m_interior = solve_thomas_algorithm(A, b)
    

#  ---------------------------------------------------------------  
    # 5. Reconstruir la solución completa (añadiendo los bordes)
    m_full = np.concatenate(([0], m_interior, [0]))
    x_full = np.linspace(x_start, x_end, N+2)
    
    return x_full, m_full



