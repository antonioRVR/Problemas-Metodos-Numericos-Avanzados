from Diferencias_finitas import solve_finite_differences
from Analisis_convergencia import convergence_analysis
import matplotlib.pyplot as plt

# --- Parámetros ---


N_test = 5 # Un número razonable de puntos
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

convergence_analysis(4000)  # Llamada a la función de análisis de convergencia

