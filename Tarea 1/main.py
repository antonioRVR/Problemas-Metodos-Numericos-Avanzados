from Diferencias_finitas import solve_finite_differences
from Analisis_convergencia import convergence_analysis
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# ---------------------------------------------------------------
#Metodo de diferencias finitas
# ---------------------------------------------------------------

# --- Parámetros ---
n=6  # Número de puntos definidos (puede modificarse)
N_test = 2*n+1 # Numero de puntos totales en la malla (incluyendo bordes y 0)


x_sol, m_sol = solve_finite_differences(N_test)

# --- Visualicemos el resultado ---
# generamos tabla de resultados
print("\n--- Tabla de Resultados ---")
print("   x         m(x)")
print("-------------------------")
for xi, mi in zip(x_sol, m_sol):
    print(f"{xi:8.5f}   {mi:12.8f}")

plt.figure(figsize=(10, 6))
plt.plot(x_sol, m_sol, 'o-', label=f'Solución Numérica (N={N_test})')
plt.title('Momento Flector Adimensional en la Barra')
plt.xlabel('Posición adimensional, x = ξ/l')
plt.ylabel('Momento flector adimensional, m(x)')
plt.grid(True)
plt.legend()
plt.show()


analis_t_f = input("¿Desea realizar el análisis de convergencia? (s/n): ")
if analis_t_f.lower() == 's':
    convergence_analysis(3001)  # Llamada a la función de análisis de convergencia
    # variar N_ref para ver como afecta a la precision del analisis. A mayor N, mejor aproximacion a la solucion exacta y el ratio de error
    # se mantiene mas proximo a 4 (orden 2 de convergencia) para mayores N en la tabla de convergencia
    # probar con N_ref = 1001, 3001, 5001
    
else:
    print("Análisis de convergencia omitido.")
    print("Fin del programa.")


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Método del disparo
# ---------------------------------------------------------------