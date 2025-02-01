import numpy as np
from tabulate import tabulate

# Инициализация начальных условий
start, end = 0.0, 0.5
initial_y = 0.0
tolerance = 0.001
max_iter = 20
epsilon = 0.001

# Функция для дифференциального уравнения
def equation(x, y):
    return (0.8-y**2)*np.cos(x)


# Метод Эйлера-Коши (приближенное решение)
def euler_method(start, end, initial_y, steps):
    step_size = (end - start) / steps
    x_points = np.linspace(start, end, steps + 1)
    y_points = [initial_y]

    for i in range(steps):
        x_i = x_points[i]
        y_i = y_points[-1]
        y_mid = y_i + step_size * equation(x_i, y_i)
        y_next = y_i + (step_size / 2) * (equation(x_i, y_i) + equation(x_i + step_size, y_mid))
        y_points.append(y_next)

    return np.array(x_points), np.array(y_points)

# Метод Рунге-Кутта 4-го порядка (точное приближение)
def runge_kutta_4(start, end, initial_y, steps):
    step_size = (end - start) / steps
    x_points = np.linspace(start, end, steps + 1)
    y_points = [initial_y]

    for i in range(steps):
        x_i = x_points[i]
        y_i = y_points[-1]
        k1 = step_size * equation(x_i, y_i)
        k2 = step_size * equation(x_i + step_size / 2, y_i + k1 / 2)
        k3 = step_size * equation(x_i + step_size / 2, y_i + k2 / 2)
        k4 = step_size * equation(x_i + step_size, y_i + k3)
        y_next = y_i + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y_points.append(y_next)

    return np.array(x_points), np.array(y_points)

# Метод двойных итераций для улучшения точности
def double_iteration(start, end, initial_y, epsilon, max_iter):
    steps = 10
    prev_x_rk, prev_y_rk = runge_kutta_4(start, end, initial_y, steps)
    prev_x_em, prev_y_em = euler_method(start, end, initial_y, steps)

    iteration = 1
    while iteration <= max_iter:
        steps *= 2
        curr_x_rk, curr_y_rk = runge_kutta_4(start, end, initial_y, steps)
        curr_x_em, curr_y_em = euler_method(start, end, initial_y, steps)

        common_indices = np.arange(0, steps + 1, 2)
        max_diff_rk = np.max(np.abs(prev_y_rk - curr_y_rk[common_indices]))
        max_diff_em = np.max(np.abs(prev_y_em - curr_y_em[common_indices]))

        if max_diff_rk < epsilon and max_diff_em < epsilon:
            table_last = []
            for i in range(-16, 0):
                table_last.append([
                    curr_x_rk[i], prev_y_rk[i // 2], curr_y_rk[i],
                    prev_y_rk[i // 2] - curr_y_rk[i], prev_y_em[i // 2], curr_y_em[i],
                    prev_y_em[i // 2] - curr_y_em[i]
                ])

            table_prev = []
            for i in range(-8, 0):
                table_prev.append([
                    prev_x_rk[i], prev_y_rk[i], curr_y_rk[2 * i], prev_y_rk[i] - curr_y_rk[2 * i],
                    prev_y_em[i], curr_y_em[2 * i], prev_y_em[i] - curr_y_em[2 * i]
                ])

            return table_prev, table_last, steps

        prev_x_rk, prev_y_rk = curr_x_rk, curr_y_rk
        prev_x_em, prev_y_em = curr_x_em, curr_y_em
        iteration += 1

    raise ValueError("Метод не сходится за максимальное кол-во итераций.")

# Запуск метода двойных итераций
table_prev, table_last, final_steps = double_iteration(start, end, initial_y, epsilon, max_iter)

# Заголовки таблицы
headers = ["x_k", "y_k (RK пред)", "y_k (RK посл)", "Δ RK", "y_k (EM пред)", "y_k (EM посл)", "Δ EM"]

# Вывод результатов
print("Методы решения дифф уравнения:")
print("RK Рунге-Кутта 4го порядка точности.")
print("EM Эйлера 2го порядка точности.\n")

print("Предпоследняя итерация:")
print(tabulate(table_prev, headers=headers, floatfmt=".3f", tablefmt="fancy_grid"))

print("\nПоследняя итерация:")
print(tabulate(table_last, headers=headers, floatfmt=".3f", tablefmt="fancy_grid"))