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