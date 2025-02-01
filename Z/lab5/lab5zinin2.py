import numpy as np
import pandas as pd

# Инициализация параметров
start, end = 0.0, 0.5
initial_y, initial_z = 0.0, 1.0
error_tolerance = 0.001


# Функция для вычисления правой части дифференциального уравнения
def equation(x, y, z):
    return (x-1)*np.sin(y)


# Однопроходный шаг метода Рунге-Кутта 4-го порядка
def rk4_step(x, y, z, step_size):
    k1y = z
    k1z = equation(x, y, z)

    k2y = z + step_size * k1z / 2
    k2z = equation(x + step_size / 2, y + step_size * k1y / 2, z + step_size * k1z / 2)

    k3y = z + step_size * k2z / 2
    k3z = equation(x + step_size / 2, y + step_size * k2y / 2, z + step_size * k2z / 2)

    k4y = z + step_size * k3z
    k4z = equation(x + step_size, y + step_size * k3y, z + step_size * k3z)

    y_next = y + (step_size / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
    z_next = z + (step_size / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)

    return y_next, z_next


# Решение системы с использованием метода Рунге-Кутта
def runge_kutta_solver(steps, step_size):
    x_vals = [start]
    y_vals = [initial_y]
    z_vals = [initial_z]

    x, y, z = start, initial_y, initial_z
    for _ in range(steps):
        x += step_size
        y, z = rk4_step(x, y, z, step_size)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)

    return x_vals, y_vals, z_vals


# Метод Адамса 3-го порядка для корректировки решения
def adams_method_3(x_vals, y_vals, z_vals, step_size, steps):
    for i in range(3, steps):
        f1 = equation(x_vals[i], y_vals[i], z_vals[i])
        f2 = equation(x_vals[i - 1], y_vals[i - 1], z_vals[i - 1])
        f3 = equation(x_vals[i - 2], y_vals[i - 2], z_vals[i - 2])

        z_next = z_vals[i] + step_size * (23 * f1 - 16 * f2 + 5 * f3) / 12
        y_next = y_vals[i] + step_size * z_next

        x_vals.append(x_vals[i] + step_size)
        y_vals.append(y_next)
        z_vals.append(z_next)

    return x_vals, y_vals


# Метод Адамса 4-го порядка для корректировки решения
def adams_method_4(x_vals, y_vals, z_vals, step_size, steps):
    for i in range(4, steps):
        f1 = equation(x_vals[i], y_vals[i], z_vals[i])
        f2 = equation(x_vals[i - 1], y_vals[i - 1], z_vals[i - 1])
        f3 = equation(x_vals[i - 2], y_vals[i - 2], z_vals[i - 2])
        f4 = equation(x_vals[i - 3], y_vals[i - 3], z_vals[i - 3])

        z_next = z_vals[i] + step_size * (55 * f1 - 59 * f2 + 37 * f3 - 9 * f4) / 24
        y_next = y_vals[i] + step_size * z_next

        x_vals.append(x_vals[i] + step_size)
        y_vals.append(y_next)
        z_vals.append(z_next)

    return x_vals, y_vals


# Двойная итерация для уточнения решений
def double_iteration_method(order=4):
    steps = 16
    prev_x = None
    prev_y = None
    results = []

    while True:
        step_size = (end - start) / steps
        x_vals, y_vals, z_vals = runge_kutta_solver(4, step_size)

        if order == 3:
            x_vals, y_vals = adams_method_3(x_vals, y_vals, z_vals, step_size, steps)
        else:
            x_vals, y_vals = adams_method_4(x_vals, y_vals, z_vals, step_size, steps)

        results.append((x_vals, y_vals))

        if prev_x is not None:
            diff = np.abs(np.interp(prev_x, x_vals, y_vals) - prev_y)
            max_diff = np.max(diff)

            if max_diff < error_tolerance:
                break

        prev_x, prev_y = x_vals, y_vals
        steps *= 2

    return prev_x, prev_y, x_vals, y_vals, results


# Получение результатов для обоих методов
x_3, y_3, last_x_3, last_y_3, result_3 = double_iteration_method(order=3)
x_4, y_4, last_x_4, last_y_4, result_4 = double_iteration_method(order=4)


# Форматирование вывода в виде таблицы
def display_table(data_frame):
    return data_frame.to_string(index=False, float_format="{:,.6f}".format)


# Создание таблиц для вывода
table_3_prev = pd.DataFrame({
    'X_k': x_3[-8:],
    'Y_k (предпоследняя)': y_3[-8:],
    'Y_k (последняя)': np.interp(x_3[-8:], last_x_3, last_y_3),
    'Разность': np.abs(y_3[-8:] - np.interp(x_3[-8:], last_x_3, last_y_3))
})

table_3_last = pd.DataFrame({
    'X_k': last_x_4[-16:],
    'Y_k (предпоследняя)': np.interp(last_x_4[-16:], x_4, y_4),
    'Y_k (последняя)': last_y_4[-16:],
    'Разность': np.abs(np.interp(last_x_4[-16:], x_4, y_4) - last_y_4[-16:])
})

table_4_prev = pd.DataFrame({
    'X_k': x_4[-8:],
    'Y_k (предпоследняя)': y_4[-8:],
    'Y_k (последняя)': np.interp(x_4[-8:], last_x_4, last_y_4),
    'Разность': np.abs(y_4[-8:] - np.interp(x_4[-8:], last_x_4, last_y_4))
})

table_4_last = pd.DataFrame({
    'X_k': last_x_4[-16:],
    'Y_k (предпоследняя)': np.interp(last_x_4[-16:], x_4, y_4),
    'Y_k (последняя)': last_y_4[-16:],
    'Разность': np.abs(np.interp(last_x_4[-16:], x_4, y_4) - last_y_4[-16:])
})

# Печать результатов
print("Адамса 3-го порядка")
print("\nпоследние 8 точек:")
print(display_table(table_3_prev))
print("\nпоследние 16 точек:")
print(display_table(table_3_last))

print("\n Адамса 4-го порядка")

print("\nпоследние 8 точек:")
print(display_table(table_4_prev))
print("\nпоследние 16 точек:")
print(display_table(table_4_last))
