# --- ANÁLISIS DE CONVERGENCIA ---

def convergence_analysis(N_ref):
    from Diferencias_finitas import solve_finite_differences
    import numpy as np
    import matplotlib.pyplot as plt
    # 1. Generar una solución de referencia muy precisa
    print("Calculando solución de referencia...")
     # Un N muy grande para simular la solución "exacta"
    x_ref, m_ref = solve_finite_differences(N_ref)
    # Obtenemos el valor en el centro, x=0, para la comparación
    # El índice del centro es N_ref // 2
    m_exact_center = m_ref[N_ref // 2 + 1] 
    print(f"Valor de referencia en x=0: {m_exact_center:.8f}")

    # 2. Definir los valores de N para la prueba
    N_values = [5, 10, 50, 100, 200, 300]
    h_values = []
    errors = []

    print("\n--- Tabla de Convergencia ---")
    print("   N      h         Error en x=0     Ratio")
    print("---------------------------------------------")

    # 3. Bucle para calcular el error para cada N
    for N in N_values:
        h = 2.0 / (N + 1)
        x_sol, m_sol = solve_finite_differences(N)
        
        # El índice del centro es N // 2
        m_center_approx = m_sol[N // 2 + 1]
        
        error = np.abs(m_center_approx - m_exact_center)
        h_values.append(h)
        errors.append(error)
        
        # Calcular el ratio de errores (excepto para el primer valor)
        ratio = errors[-2] / errors[-1] if len(errors) > 1 else 0.0
        
        print(f"{N:4d}   {h:8.5f}   {error:e}   {ratio:5.2f}")

    # 4. Generar la gráfica log-log y calcular la pendiente
    h_values = np.array(h_values)
    errors = np.array(errors)

    # Ajuste lineal en escala logarítmica para encontrar la pendiente
    # log(error) = log(C) + p * log(h)  <-- p es la pendiente (orden de convergencia)
    coeffs = np.polyfit(np.log(h_values), np.log(errors), 1)
    slope = coeffs[0]

    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors, 'o-', label='Error Numérico')
    plt.title(f'Análisis de Convergencia (Pendiente = {slope:.2f})')
    plt.xlabel('Tamaño del paso, h')
    plt.ylabel('Error absoluto en x=0')
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis() # Es común mostrar h decreciendo
    plt.legend()
    plt.show()
