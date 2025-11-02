# Código principal para llamar a los metodos numericos.
# permite elegir el metodo a utilizar y si se desea realizar el analisis de convergencia


from Diferencias_finitas import solve_finite_differences
from Analisis_convergencia import convergence_analysis
import matplotlib.pyplot as plt


print("Que problema desea resolver?")
print("1: Problema de contorno usando el método de diferencias finitas.")
print("2: Problema de contorno usando el método del disparo.")
print("3: salir.")

opcion = input("Ingrese 1, 2 o 3: ")

# --- Parámetros ---

n=7  # Número de puntos definidos (puede modificarse)
N_test = 2*n-1 # Numero de puntos interiores en la malla (incluyendo 0)

while opcion != "3":
        
    if opcion == '1':
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        #Metodo de diferencias finitas
        # ---------------------------------------------------------------

        x_sol, m_sol = solve_finite_differences(N_test)

        # generamos tabla de resultados
        print("\n--- Tabla de Resultados ---")
        print("   x         m(x)")
        print("-------------------------")
        for xi, mi in zip(x_sol, m_sol):
            print(f"{xi:8.5f}   {mi:12.8f}")

        plt.figure(figsize=(10, 6))
        plt.plot(x_sol, m_sol, 'o-', label=f'(N*={N_test+2})') # +2 para incluir los bordes
        plt.title('Momento Flector en la Barra')
        plt.xlabel('Posición adimensional, x = ξ/l')
        plt.ylabel('Momento flector adimensional, m(x)')
        plt.grid(True)
        plt.legend()
        plt.show()


        analis_t_f = input("¿Desea realizar el análisis de convergencia? (s/n): ")
        if analis_t_f.lower() == 's':
            convergence_analysis(3001, opcion)  # Llamada a la función de análisis de convergencia
            # variar N_ref para ver como afecta a la precision del analisis. A mayor N, mejor aproximacion a la solucion exacta y el ratio de error
            # se mantiene mas proximo a 4 (orden 2 de convergencia) para mayores N en la tabla de convergencia
            # probar con N_ref = 1001, 3001, 5001
            
        else:
            print("Análisis de convergencia omitido.")
            print("Fin del programa.")
        
        print("\nQue problema desea resolver?")
        print("1: Problema de contorno usando el método de diferencias finitas.")
        print("2: Problema de contorno usando el método del disparo.")
        print("3: salir.")

        opcion = input("Ingrese 1, 2 o 3: ")
        continue

    elif opcion == '2':

        # --------------------------------- ------------------------------
        # ---------------------------------------------------------------
        # Método del disparo
        # ---------------------------------------------------------------

        from Metodo_disparo import solve_linear_shooting
        x_sol, m_sol = solve_linear_shooting(n)
        # --- Visualicemos el resultado ---
        plt.figure(figsize=(10, 6))
        plt.plot(x_sol, m_sol, 'o-', label=f'(N*={N_test+2})') # +2 para incluir los bordes
        plt.title('Momento Flector en la Barra (Método del Disparo)')
        plt.xlabel('Posición adimensional, x')
        plt.ylabel('Momento flector adimensional, m(x)')
        plt.grid(True)
        plt.legend()
        plt.show()


        print("\n--- Tabla de Resultados (Método del Disparo) ---")
        print(f" {'x':^7}  {'m(x)':^15} ")
        print(" -------------------------")
        for xi, mi in zip(x_sol, m_sol): 
            print(f"{xi:8.5f}   {mi:12.8f}")


        analis_t_f = input("¿Desea realizar el análisis de convergencia? (s/n): ")
        if analis_t_f.lower() == 's':
            convergence_analysis(3001, opcion)  # Llamada a la función de análisis de convergencia
            # variar N_ref para ver como afecta a la precision del analisis. A mayor N, mejor aproximacion a la solucion exacta y el ratio de error
            # se mantiene mas proximo a 4 (orden 2 de convergencia) para mayores N en la tabla de convergencia
            # probar con N_ref = 1001, 3001, 5001
            
        else:
            print("Análisis de convergencia omitido.")
            print("Fin del programa.")

        print("Fin del programa.")

    print("\nQue problema desea resolver?")
    print("1: Problema de contorno usando el método de diferencias finitas.")
    print("2: Problema de contorno usando el método del disparo.")
    print("3: salir.")
    opcion = input("Ingrese 1, 2 o 3: ")
    continue