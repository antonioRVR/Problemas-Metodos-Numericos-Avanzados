# este codigo realiza la preparación del problema de diferencias finitas.
# depnde del codigo Matrix_solver.py para resolver el sistema lineal tridiagonal

def solve_finite_differences(N):
    from Matrix_solver import solve_crout
    import numpy as np

    # 1. Definir el dominio y la malla de puntos
    x_start, x_end = -1.0, 1.0
    h = (x_end - x_start) / (N + 1) 


    # Puntos interiores de la malla
    x_interior = np.linspace(x_start + h, x_end - h, N)


    # 2. Construir la matriz A (tamaño (N) x (N))
    # Diagonal principal
    diag_main = h**2 * (1 + x_interior**2) - 2
    
    # Diagonales superior e inferior (todo unos)
    diag_upper = np.ones(N - 1)
    diag_lower = np.ones(N - 1)
    

    # Ensamblar la matriz tridiagonal
    A = np.diag(diag_main) + np.diag(diag_upper, k=1) + np.diag(diag_lower, k=-1)

    # 3. Construir el vector b
    b = -h**2 * np.ones(N)
    
#  ---------------------------------------------------------------  
#  --------------------------------------------------------------- 
# Resolver el sistema lineal A * m = b
#  ---------------------------------------------------------------  

# A opcion simple con numpy
    # m_interior = np.linalg.solve(A, b)

# B opcion con algoritmo de Crout implementado. ideal para matrices tridiagonales. referenciado de bibligrafia
    m_interior = solve_crout(A, b)

#  ---------------------------------------------------------------  
#  ---------------------------------------------------------------  
# Reconstruir la solución completa (añadiendo los bordes)
#  ---------------------------------------------------------------  

    m_full = np.concatenate(([0], m_interior, [0]))
    x_full = np.linspace(x_start, x_end, N+2)
 
    return x_full, m_full

