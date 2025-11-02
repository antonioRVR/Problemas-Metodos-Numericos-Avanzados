# --- ANÁLISIS DE CONVERGENCIA ---

#este codigo realiza un análisis de convergencia para ambos metodos numericos.
# depende del parametro 'problem' para decidir cual metodo analizar: "1" para diferencias finitas, "2" para metodo del disparo

def convergence_analysis(N_ref, problem):
    from Diferencias_finitas import solve_finite_differences
    from Metodo_disparo import solve_linear_shooting
    import numpy as np
    import matplotlib.pyplot as plt

    # ----------------------------------------------------
    # Análisis de convergencia para diferencias finitas
    # ----------------------------------------------------
    if problem == "1":
        print("Análisis de convergencia para el método de diferencias finitas.")
        # 1. Generar una solución de referencia 
        print("Calculando solución de referencia")
        # Un N muy grande para simular la solución exacta
        x_ref, m_ref = solve_finite_differences(N_ref)

        # Obtenemos el valor en el centro, x=0, para la comparación
        # El índice del centro es N_ref // 2
        m_exact_center = m_ref[N_ref // 2 + 1] 
        print(f"Valor de referencia en x=0: ,{m_exact_center:.8f}")

        # 2. Definir los valores de N para la prueba
        N_values = [5, 11, 21, 41, 81, 161, 321, 641,1281]

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
        coeffs = np.polyfit(np.log(h_values), np.log(errors), 1)
        slope = coeffs[0]

        plt.figure(figsize=(10, 6))
        plt.loglog(h_values, errors, 'o-', label='Error Numérico')
        plt.title(f'Análisis de Convergencia (Pendiente = {slope:.2f})')
        plt.xlabel('Tamaño del paso, h')#
        plt.ylabel('Error absoluto en x=0')
        plt.grid(True, which="both", ls="--")
        plt.gca()

        plt.legend()
        plt.show()


    # ----------------------------------------------------
    # Análisis de convergencia para metodo de disparo
    # ----------------------------------------------------
    elif problem == "2":

        print("Análisis de convergencia para el método del disparo.")
        # 1. Generar una solución de referencia
        print("Calculando solución de referencia")
        x_ref, m_ref = solve_linear_shooting(N_ref)

        # Obtenemos el valor en el centro, x=0
        # Con N_enunciado, el índice del centro (x=0) es N_ref_enunciado
        m_exact_center = m_ref[N_ref]
        print(f"Valor de referencia en x=0: ,{m_exact_center:.8f}")

        # 2. Definir los valores de comparacion
        N_values = [5, 11, 21, 41, 81, 161, 321, 641,1281]

        h_values = []
        errors = []

        print("\n--- Tabla de Convergencia ---")
        print("   N      h         Error en x=0     Ratio")
        print("---------------------------------------------")

        # 3. Bucle para calcular el error para cada N
        for N in N_values:
            h = 1.0 / N
            x_sol, m_sol = solve_linear_shooting(N)
            
            # El índice del centro es
            m_center_approx = m_sol[N]
            
            error = np.abs(m_center_approx - m_exact_center)
            h_values.append(h)
            errors.append(error)
            
            # Calcular el ratio de errores
            ratio = errors[-2] / errors[-1] if len(errors) > 1 else 0.0
            
            print(f"{N:9d}   {h:8.5f}   {error:e}   {ratio:5.2f}")

        # 4. Generar la gráfica log-log y calcular la pendiente
        h_values = np.array(h_values)
        errors = np.array(errors)

        coeffs = np.polyfit(np.log(h_values), np.log(errors), 1)
        slope = coeffs[0]

        plt.figure(figsize=(10, 6))
        plt.loglog(h_values, errors, 'o-', label='Error Numérico')
        plt.title(f'Análisis de Convergencia (Pendiente = {slope:.2f})')
        plt.xlabel('Tamaño del paso, h')
        plt.ylabel('Error absoluto en x=0')
        plt.grid(True, which="both", ls="--")
        plt.gca()
        plt.legend()
        plt.show()
