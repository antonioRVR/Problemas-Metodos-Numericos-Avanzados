# este codigo implementa el metodo del disparo lineal
#  y la resolución por runge kutta orden 4

import numpy as np
import matplotlib.pyplot as plt

# 1. Definimos las funciones f1 y f2 para el sistema ---

def f1(x, y):
    
    # Sistema de EDOs para el Problema 1 (no homogéneo).
    # y es el vector [m, m'] = [y[0], y[1]]
    
    m = y[0]
    v = y[1] # v = m'
    dmdx = v
    dvdx = -(1 + x**2) * m - 1
    return np.array([dmdx, dvdx])

def f2(x, y):

    # Sistema de EDOs para el Problema 2 (homogéneo).
    # y es el vector [m, m'] = [y[0], y[1]]

    m = y[0]
    v = y[1] # v = m'
    dmdx = v
    dvdx = -(1 + x**2) * m
    return np.array([dmdx, dvdx])

# 2. Implementamos RK4 genérico ---

def solve_with_rk4(f, y0, x_range, h):
    # Resuelve un sistema de EDOs dy/dx = f(x, y) usando RK4.
    # f: función que define el sistema (f1 o f2).
    # y0:  vector de condiciones iniciales [m(a), m'(a)].
    # x_range: Una tupla (x_start, x_end).
    # h: El tamaño del paso.
    
    x_start, x_end = x_range
    

    # ----------------
    # Usar solo si tenemos problemas de precisión numérica al final de la malla. 

    # Usamos np.arange para asegurar que el último punto no se exceda
    # y añadimos un pequeño epsilon a x_end para incluirlo si es exacto
    # para evitar problemas de precisión numérica al final de la malla 
    # debido a como python calcula np.arange. 
    #x_points = np.arange(x_start, x_end + h*0.5, h)
    # ---------------

    x_points = np.arange(x_start, x_end + h, h)

    N_steps = len(x_points) - 1
    
    # Array para guardar la solución. y_sol[i] será el vector [m, m'] en x_i
    y_sol = np.zeros((N_steps + 1, len(y0)))
    y_sol[0] = y0
    
    # Bucle principal de RK4 
    # con esto iteramos sobre cada paso de la malla 
    for i in range(N_steps):
        xi = x_points[i]
        yi = y_sol[i]
        
        # Fórmulas del método RK4
        k1 = h * f(xi, yi)
        k2 = h * f(xi + h/2, yi + k1/2)
        k3 = h * f(xi + h/2, yi + k2/2)
        k4 = h * f(xi + h, yi + k3)
        
        y_sol[i+1] = yi + (k1 + 2*k2 + 2*k3 + k4) / 6
        
    # Devolvemos los puntos x y el array de soluciones [m, m']
    return x_points, y_sol

# 3. Implementamos la lógica del Método del Disparo ---

def solve_linear_shooting(N_enunciado):
    
    # Resuelvemos el problema de contorno usando el Método del Disparo Lineal.
    #  dominio y paso
    x_range = (-1.0, 1.0)
    h = 1.0 / N_enunciado 

    # 2. Resolver el Problema de Valor Inicial 1
    y0_1 = np.array([0.0, 0.0]) # [m₁(-1), m₁'(-1)]
    x, y1_solution_full = solve_with_rk4(f1, y0_1, x_range, h)
    # y1_solution_full es un array de vectores [m, m']. 

    # Nos quedamos solo con la columna 'm' (índice 0)
    #  porque es la que nos interesa para la solución final 
    #  las otras columnas son las derivadas m' que no necesitamos devolver 
    m1 = y1_solution_full[:, 0]

    # 3. Resolver el Problema de Valor Inicial 2
    y0_2 = np.array([0.0, 1.0]) # [m₂(-1), m₂'(-1)]
    # No necesitamos guardar x de nuevo, ya que es el mismo
    _, y2_solution_full = solve_with_rk4(f2, y0_2, x_range, h)
    m2 = y2_solution_full[:, 0]

    # 4. Encontrar la constante C
    # C = -m₁(1) / m₂(1)
    C = -m1[-1] / m2[-1] # -1 es el índice del último elemento (en x=1)

    # 5. Calcular la solución final como suma de las dos soluciones
    m_final = m1 + C * m2
    
    return x, m_final