import pandas as pd
import numpy as np
import lstsq_classifier


def main():
    print("Executing...")


if __name__ == "__main__":
    main()

def small_matrix_ex():
    # Definir la matriz 15x10
    A_ex = np.zeros((15, 5))

    # Asignar valores aleatorios a las columnas 1-3 dentro del rango 150000 a 400000
    A_ex[:, 0:3] = np.random.uniform(150000, 400000, size=(15, 3))

    # Asignar valores aleatorios al resto de columnas entre 0 y 1
    A_ex[:, 3:] = np.random.uniform(0, 1, size=(15, 2))

    # Crear un vector con 15 componentes entre 1 y 4 (enteros)
    b_ex = np.random.randint(1, 5, size=15)

    #####

    # Mostrar albumes con sus clases asignadas previamente
    print("\nDATOS DE ENTRADA (reales):")
    df = pd.DataFrame({'Album': np.arange(1, 16), 'Vector': b_ex})
    print(df)

    print("\n\nDATOS OBTENIDOS POR EL MODELO:")

    n = 4
    b_results = []
    b_results_no_poly = []

    print("\n\nModelo no polinomial")
    b_result_no_poly, p_result_no_poly, x_result_no_poly = lstsq_classifier.least_squares_multiclass_classifier(A_ex, b_ex, 4)
    b_results_no_poly.append(b_result_no_poly)

    for j in range(1, n + 1):
        print(f"\n\nModelo polinomial de grado {j}")
        b_result, p_result, x_result = lstsq_classifier.lstsq_classifier_poly(A_ex, b_ex, 4, j)
        b_results.append(b_result)


