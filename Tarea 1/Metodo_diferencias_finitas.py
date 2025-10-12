import numpy as np
import matplotlib.pyplot as plt

def solve_finite_differences(N):
    """
    Resuelve el problema de contorno usando el método de diferencias finitas.
    """
    # 1. Definir el dominio y la malla
    x_start, x_end = -1.0, 1.0
    h = (x_end - x_start) / N
    # Puntos interiores de la malla (donde m es desconocido)
    x_interior = np.linspace(x_start + h, x_end - h, N - 1)
    
    # 2. Construir la matriz A (tamaño (N-1) x (N-1))
    # Diagonal principal
    diag_main = h**2 * (1 + x_interior**2) - 2
    
    # Diagonales superior e inferior (todo unos)
    diag_upper = np.ones(N - 2)
    diag_lower = np.ones(N - 2)
    
    # Ensamblar la matriz tridiagonal
    A = np.diag(diag_main) + np.diag(diag_upper, k=1) + np.diag(diag_lower, k=-1)
    
    # 3. Construir el vector b
    b = -h**2 * np.ones(N - 1)
    
    # 4. Resolver el sistema lineal A * m = b
    m_interior = np.linalg.solve(A, b)
    
    # 5. Reconstruir la solución completa (añadiendo los bordes)
    m_full = np.concatenate(([0], m_interior, [0]))
    x_full = np.linspace(x_start, x_end, N + 1)
    
    return x_full, m_full

# --- Probemos el código por primera vez ---
N_test = 50 # Un número razonable de puntos
x_sol, m_sol = solve_finite_differences(N_test)

# --- Visualicemos el resultado ---
plt.figure(figsize=(10, 6))
plt.plot(x_sol, m_sol, 'o-', label=f'Solución Numérica (N={N_test})')
plt.title('Momento Flector Adimensional en la Barra')
plt.xlabel('Posición adimensional, x = ξ/l')
plt.ylabel('Momento flector adimensional, m(x)')
plt.grid(True)
plt.legend()
plt.show()