import pandas as pd
import numpy as np

def least_squares_multiclass_classifier(A, b, k):
    # K = cantidad de clases, para cada una se realiza un 'lstsq'
    # binario como lo propone boyd

    # Inicializar matrices de coeficientes y probabilidades
    x = np.zeros((A.shape[1], k))
    p = np.zeros((A.shape[0], k))

    # Para cada clase en el conjunto E
    for i in range(1, k + 1):
        # Crear un vector de etiquetas binarias bi (m x 1)
        # tomamos el b pasado como parámetro, hacemos una comparación
        # para que así quede solo con dos posibles valores: 1 o 0
        bi = np.where(b == i, 1, 0)

        # Resolver el sistema de ecuaciones A xi = bi usando lstsq
        # resolvemos esa clasficación binaria
        xi = np.linalg.lstsq(A, bi, rcond=None)[0]

        # Guardar el vector de coeficientes xi (n x 1) en la matriz x
        # : significa en cada fila y (i-1) porque i inicia desde 1, las
        # posiciones en python inician desde 0 (x es una matriz con las n filas
        # como características hallan y m columnas como clases hallan), es decir,
        # para cada clase tengo un vector 'xi' (que cubren todas las características - coeficientes)
        x[:, i - 1] = xi[:A.shape[1]]
        # obtenemos los primeros m coeficientes (correspondientes a las características)

        # Calcular el vector de probabilidades pi (m x 1) usando la función sigmoide
        # teniendo el vector xi para cada clase para todos los datos, hago
        # producto entre A y xi para saber la clase que dice el modelo a la cual
        # pertenece cada dato y convierto en probabilidad cada componente
        pi = 1 / (1 + np.exp(-A @ xi[:A.shape[1]]))

        # Guardar el vector de probabilidades pi en la matriz p
        # guardamos la probabilidad que tiene cada dato de que pertenezca
        # a cada una de las clases. P tiene como columnas la cantidad de clases
        # y como filas la cantidad de datos
        p[:, i - 1] = pi

        b_result = np.zeros((A.shape[0], 1))

    # Imprimir las predicciones
    for j in range(p.shape[0]):
      # argmax me devuelve el índice, estamos escogiendo para un dato (una fila)
      # el mayor valor que hay entre las 4 clases. Sumamos 1 para que ya tengamos la
      # clase a la que pertenece ya que Python indexa desde 0
      max_index = np.argmax(p[j, :]) + 1
      b_result[j] = max_index
      print(f"El álbum {j + 1} pertenece a la clase {max_index} con una probabilidad de {p[j, max_index - 1]:.2f}")

    return b_result, p, x # prueba -> devolver a x, contiene k vectores xi
    # que indican los coeficientes de cada clase para cada caracteristica


def generate_polynomial_features(A, degree):
    if degree + 1 > A.shape[1]:
        return 0
    else:
        # Creamos una nueva matriz con la primera columna llena de unos
        nueva_matriz = np.zeros((A.shape[0], degree + 1))

        # Llenamos las columnas restantes
        for i in range(0, degree+1):
            nueva_matriz[:, i] = np.ravel(A[:, i]) ** i

        return nueva_matriz


def lstsq_classifier_poly(A, b, k, degree):
    # Agregar términos polinómicos a la matriz de características
    A_poly = generate_polynomial_features(A, degree)
    print('A poly: ', A_poly)
    return least_squares_multiclass_classifier(A_poly, b, k)