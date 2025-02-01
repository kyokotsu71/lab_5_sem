import numpy as np

def f(x):
    return x - 3 * np.cos(x)**2

def f_prime(x):
    return 1 + 6 * np.cos(x) * np.sin(x)


def chord_method(x0, x1, tolerance=1e-8, max_iterations=100):
    for i in range(max_iterations):
        # Вычисляем значение функции в двух точках
        f_x0 = f(x0)
        f_x1 = f(x1)

        # Находим новую точку по формуле хорд
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        li.append(round(x_new, 8))
        # Проверяем условие остановки
        if abs(x_new - x1) < tolerance:
            return round(x_new, 8)

        # Обновляем точки
        x0, x1 = x1, x_new

    raise ValueError("Метод не сошелся")


def newton_method(x0, tolerance=1e-8, max_iterations=100):
    x = x0
    for i in range(max_iterations):
        x_new = x - f(x) / f_prime(x)
        if f_prime(x) == 0:
            raise ValueError("производная нолик")

        print(f'iter {i+1} x = {x_new:.8f}, |xnew-x| = {abs(x_new-x):.8f}')
        if abs(x_new - x) < tolerance:
            return round(x_new, 8)
        x = x_new
    raise ValueError("не сошелся")


def secant_method(x0, x1, tolerance=1e-8, max_iterations=100):
    for i in range(max_iterations):
        # Вычисляем значение функции в двух точках
        f_x0 = f(x0)
        f_x1 = f(x1)

        # Находим новую точку по формуле секущих
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        li.append(f'{x_new:.8f}')
        # Проверяем условие остановки
        if abs(x_new - x1) < tolerance:
            return round(x_new, 8)

        # Обновляем точки
        x0, x1 = x1, x_new

    raise ValueError("Метод не сошелся")

def newton_method_2(x0, h, tolerance=1e-8, max_iterations=100):
    for i in range(max_iterations):
        # Вычисляем значение функции в двух точках
        f_x0 = f(x0)

        # Находим новую точку по формуле секущих
        x_new = x0 - h * f_x0/(f(x0+h) - f_x0)
        li.append(f'{x_new:.8f}')
        # Проверяем условие остановки
        if abs(x_new - x0) < tolerance:
            return round(x_new, 8)

        # Обновляем точки
        x0 = x_new

    raise ValueError("Метод не сошелся")





def steff_method(x0, tolerance=1e-8, max_iterations=100):
    for i in range(max_iterations):
        # Вычисляем значение функции в двух точках
        f_x0 = f(x0)

        # Находим новую точку по формуле секущих
        x_new = x0 - f_x0**2/(f(x0+f_x0)-f_x0)
        li.append(f'{x_new:.8f}')
        # Проверяем условие остановки
        if abs(x_new - x0) < tolerance:
            return round(x_new, 8)

        # Обновляем точки
        x0 = x_new

    raise ValueError("Метод не сошелся")



def iter_method(x0, t, tolerance=1e-8, max_iterations=100):
    for i in range(max_iterations):
        f_x0 = f(x0)

        x_new = x0-t*f_x0
        li.append(f'{x_new:.8f}')
        if abs(x_new - x0) < tolerance:
            return round(x_new, 8)

        x0 = x_new

    raise ValueError("Метод не сошелся")


x0 = 1.0  # Начальная точка
t = 0.5
li = [f'{x0:.8f}']
iter_method(x0, t)
print(li)