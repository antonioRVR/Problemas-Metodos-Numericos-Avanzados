# este codigo implementa el algoritmo de Crout para resolver sistemas de ecuaciones lineales con matrices tridiagonales.

def solve_crout(A_matrix, b_vector):
    import numpy as np


    # Resuelve un sistema tridiagonal Ax=b 
    # A_matrix: La matriz tridiagonal completa
    # b_vector: El vector de términos independientes 
    # x: El vector de la solución.
    
    # tamaño del sistema
    n = len(b_vector)
    
    # Extraemos las diagonales de la matriz A
    a = np.diag(A_matrix, k=-1) # Diagonal inferior (a_i empieza en i=2)
    b = np.diag(A_matrix, k=0)  # Diagonal principal (b_i empieza en i=1)
    c = np.diag(A_matrix, k=1)  # Diagonal superior (c_i empieza en i=1)

    # Creamos los vectores l, u, z del algoritmo
    # y los inicializamos a cero
    l = np.zeros(n)
    u = np.zeros(n - 1) # u solo tiene n-1 elementos
    z = np.zeros(n)
    
    # Paso 1: Descomposición y sustitución hacia adelante
    
    l[0] = b[0]
    if n > 1:
        u[0] = c[0] / l[0]
    
    for i in range(1, n - 1):
        # El libro usa a[i] para la diagonal inferior, 
        # pero el índice en Python de 'a' es i-1
        l[i] = b[i] - a[i-1] * u[i-1]
        u[i] = c[i] / l[i]
        
    # Último elemento de l
    if n > 1:
        l[n-1] = b[n-1] - a[n-2] * u[n-2]
    
    # Paso 2: Resolver Lz = b 
    z[0] = b_vector[0] / l[0]
    for i in range(1, n):
        # El índice de 'a' en Python es i-1
        z[i] = (b_vector[i] - a[i-1] * z[i-1]) / l[i]
        
    # Paso 3: Sustitución hacia atrás (Resolver Ux = z)
    x = np.zeros(n)
    x[n-1] = z[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = z[i] - u[i] * x[i+1]
        
    return x