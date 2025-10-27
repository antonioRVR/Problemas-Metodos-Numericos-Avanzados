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

# # Example usage:  
# if __name__ == "__main__":
#     # Example system:
#     # 2x + y = 5
#     # 3x + 2y = 8
#     A = [[2, 1],
#          [3, 2]]
#     b = [5, 8]
    
#     solution = Solve_Gaussian_el(A, b)
#     print("Solution:", solution)

def Solve_Gaussian_el(A, b):
    import numpy as np
    """
    Resuelve un sistema de ecuaciones lineales Ax=b usando
    eliminación Gaussiana sin pivoteo.
    """
    N = len(b)
    
    # Creamos una copia para no modificar la matriz y el vector originales
    A_copy = np.copy(A).astype(float)
    b_copy = np.copy(b).astype(float)
    
    # --- 1. Fase de Eliminación hacia adelante ---
    # El objetivo es convertir A en una matriz triangular superior
    for i in range(N):
        # Para cada fila 'i', eliminamos la variable x_i de las filas de abajo
        for j in range(i + 1, N):
            # Calculamos el factor por el que multiplicar la fila 'i'
            # para anular el primer elemento de la fila 'j'
            factor = A_copy[j, i] / A_copy[i, i]
            
            # Actualizamos la fila 'j' de la matriz A
            A_copy[j, i:] = A_copy[j, i:] - factor * A_copy[i, i:]
            
            # Actualizamos el elemento correspondiente en el vector b
            b_copy[j] = b_copy[j] - factor * b_copy[i]
            
    # --- 2. Fase de Sustitución hacia atrás ---
    # Ahora resolvemos el sistema triangular superior resultante
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        # Empezamos desde la última ecuación y vamos hacia arriba
        suma = np.dot(A_copy[i, i+1:], x[i+1:])
        x[i] = (b_copy[i] - suma) / A_copy[i, i]
        
    return x


def solve_thomas_algorithm(A, b):
    import numpy as np
    """
    Resuelve un sistema de ecuaciones lineales tridiagonal Ax=b
    usando el Algoritmo de Thomas. A debe ser la matriz completa.
    """
    N = len(b)
    # Extraer las diagonales de A
    c = np.diag(A, k=1)  # Diagonal superior
    d = np.diag(A, k=0)  # Diagonal principal
    a = np.diag(A, k=-1) # Diagonal inferior
    
    # Copiamos para no modificar los originales
    c_prime = np.copy(c)
    d_prime = np.copy(d)
    b_prime = np.copy(b)
    
    # 1. Fase de eliminación hacia adelante
    for i in range(1, N):
        m = a[i-1] / d_prime[i-1]
        d_prime[i] = d_prime[i] - m * c_prime[i-1]
        b_prime[i] = b_prime[i] - m * b_prime[i-1]
        
    # 2. Fase de sustitución hacia atrás
    x = np.zeros(N)
    x[-1] = b_prime[-1] / d_prime[-1]
    for i in range(N - 2, -1, -1):
        x[i] = (b_prime[i] - c_prime[i] * x[i+1]) / d_prime[i]
        
    return x


def solve_crout(A,b):
    import numpy as np
    """
    Resuelve un sistema de ecuaciones lineales Ax=b usando
    la descomposición LU de Crout.
    """
    N = len(b)
    # Crear matrices L y U
    L = np.zeros((N, N))
    U = np.zeros((N, N))
    
    # Descomposición LU
    for j in range(N):
        L[j, 0] = A[j, 0]
    for i in range(N):
        U[i, i] = 1.0
    
    for j in range(1, N):
        for i in range(j, N):
            sum1 = sum(L[i, k] * U[k, j] for k in range(j))
            L[i, j] = A[i, j] - sum1
        for i in range(j + 1, N):
            sum2 = sum(L[j, k] * U[k, i] for k in range(j))
            U[j, i] = (A[j, i] - sum2) / L[j, j]
    
    # Resolver Ly=b
    y = np.zeros(N)
    for i in range(N):
        sum3 = sum(L[i, k] * y[k] for k in range(i))
        y[i] = (b[i] - sum3) / L[i, i]
    
    # Resolver Ux=y
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        sum4 = sum(U[i, k] * x[k] for k in range(i + 1, N))
        x[i] = y[i] - sum4
    
    return x

def solve_crout_2(A_matrix, b_vector):
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