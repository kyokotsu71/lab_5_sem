import numpy as np

def gauss(A, b):
    n = len(b)
    # прямой ход
    for i in range(n):
        # нормализация строки
        max = np.argmax(np.abs(A[i:, i])) + i
        A[[i, max]] = A[[max, i]]  # меняем строки
        b[i], b[max] = b[max], b[i]  # меняем элементы в b

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j][i:] = A[j][i:] - factor * A[i][i:]
            b[j] = b[j] - factor * b[i]

    # обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return x, A



# система
A = np.array([[3.82, 1.02, 0.75, 0.81],
                     [1.05, 4.53, 0.98, 1.53],
                     [0.73, 0.85, 4.71, 0.81],
                     [0.88, 0.81, 1.28, 3.50]])
b = np.array([16.855, 22.705, 22.480, 16.110])
x_true = np.array([2.5, 3.0, 3.5, 2.0])
x, solution = gauss(A, b)
error = np.linalg.norm(x - x_true)
print('матрица:')
print(solution)
print('решение: ', x)
print('погрешность: ', error)




import numpy as np


def gauss_seidel(A, b, x0=None, max_iterations=100, tolerance=1e-10):
    n = len(b)
    x = np.zeros(n)

    matrix = np.zeros_like(A)
    vector = np.zeros(n)

    for i in range(n):
        matrix[i][:]= -A[i][:]/A[i][i]
        matrix[i][i] = 0
        vector[i]=b[i]/A[i][i]

    print(matrix)

    for iteration in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(matrix[i, :i], x[:i])
            sum2 = np.dot(matrix[i, i + 1:], x_old[i + 1:])
            x[i] = vector[i]+sum1 +sum2

    # Проверка на сходимость
        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            print(f"всего {iteration + 1} итераций")
            return x, iteration + 1

def calculate_error(true_solution, computed_solution):
    return np.linalg.norm(true_solution - computed_solution)


A = np.array([[3.82, 1.02, 0.75, 0.81],
              [1.05, 4.53, 0.98, 1.53],
              [0.73, 0.85, 4.71, 0.81],
              [0.88, 0.81, 1.28, 3.50]])
b = np.array([16.855, 22.705, 22.480, 16.110])
x_true = np.array([2.5, 3.0, 3.5, 2.0])

x_seidel, iterations = gauss_seidel(A, b)
error = calculate_error(x_true, x_seidel)
print("решение:")
print(x_seidel)
print("всего итераций: ", iterations)
print('погрешность: ', error)

