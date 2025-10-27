def Solve_Gaussian_e_0(A, b):
    """
    Solve the system of linear equations Ax = b using Gaussian elimination
    with partial pivoting.
    
    Parameters:
    A : list of lists representing the coefficient matrix
    b : list representing the right-hand side vector
    
    Returns:
    x : list representing the solution vector
    """
    n = len(A)
    # Create augmented matrix [A|b]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for i in range(n):
        # Partial pivoting
        pivot = abs(M[i][i])
        pivot_row = i
        for j in range(i + 1, n):
            if abs(M[j][i]) > pivot:
                pivot = abs(M[j][i])
                pivot_row = j
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]
            
        # Make all rows below this one 0 in current column
        for j in range(i + 1, n):
            factor = M[j][i] / M[i][i]
            for k in range(i, n + 1):
                M[j][k] -= factor * M[i][k]
    
    # Back substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    
    return x

def solve_crout(A_matrix, b_vector):
    import numpy as np
    """
    Resuelve un sistema tridiagonal Ax=b usando el Algoritmo 6.7 de 
    Burden y Faires (Factorización de Crout / Algoritmo de Thomas).

    Argumentos:
    A_matrix: La matriz tridiagonal completa (NxN).
    b_vector: El vector de términos independientes (Nx1).

    Devuelve:
    x: El vector de la solución.
    """
    
    # Extraemos el tamaño del sistema
    n = len(b_vector)
    
    # Extraemos las diagonales de la matriz A
    # El libro las llama a, b, c (inferior, principal, superior)
    a = np.diag(A_matrix, k=-1) # Diagonal inferior (a_i empieza en i=2)
    b = np.diag(A_matrix, k=0)  # Diagonal principal (b_i empieza en i=1)
    c = np.diag(A_matrix, k=1)  # Diagonal superior (c_i empieza en i=1)

    # Creamos los vectores l, u, z del algoritmo
    # (Los inicializamos con ceros)
    l = np.zeros(n)
    u = np.zeros(n - 1) # u solo tiene n-1 elementos
    z = np.zeros(n)
    
    # --- Paso 1: Descomposición y sustitución hacia adelante ---
    # Sigue exactamente los pasos del Algoritmo 6.7
    
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
    
    # --- Paso 2: Resolver Lz = b ---
    z[0] = b_vector[0] / l[0]
    for i in range(1, n):
        # El índice de 'a' en Python es i-1
        z[i] = (b_vector[i] - a[i-1] * z[i-1]) / l[i]
        
    # --- Paso 3: Sustitución hacia atrás (Resolver Ux = z) ---
    x = np.zeros(n)
    x[n-1] = z[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = z[i] - u[i] * x[i+1]
        
    return x