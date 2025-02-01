import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
h = 0.1
tau = 0.005
D = 1
a, b = 0, 1
c, d = 0, 10

# Определение количества узлов
I = int((b - a) / h) + 1
J = int((d - c) / tau) + 1

# Создание сетки
x = np.linspace(a, b, I)
t = np.linspace(c, d, J)
U = np.zeros((J, I))

# Начальные условия
U[0, :] = x ** 4
U[:, 0] = 0
U[:, -1] = 1

# Параметр λ
lambda_ = (D * tau) / (h ** 2)
if lambda_ >= 0.5:
    raise ValueError("λ должно быть меньше 0.5.")

# Явный метод
for j in range(0, J - 1):
    for i in range(1, I - 1):
        U[j + 1, i] = lambda_ * U[j, i + 1] + (1 - 2 * lambda_) * U[j, i] + lambda_ * U[j, i - 1]

# Визуализация результатов
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(T, X, U, cmap='plasma')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('U')
ax.set_title('явный')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
h = 0.1
tau = 0.005
D = 1
a, b = 0, 1
c, d = 0, 10

# Определение количества узлов
I = int((b - a) / h) + 1
J = int((d - c) / tau) + 1

# Создание сетки
x = np.linspace(a, b, I)
t = np.linspace(c, d, J)
U = np.zeros((J, I))

# Начальные условия
U[0, :] = x ** 4
U[:, 0] = 0
U[:, -1] = 1

# Параметр λ
lambda_ = D * tau / h ** 2
if lambda_ >= 0.5:
    raise ValueError("Условие устойчивости не выполнено: λ должно быть меньше 0.5.")

# Неявный метод
for j in range(0, J - 1):
    alpha = np.zeros(I)
    beta = np.zeros(I)

    for i in range(1, I - 1):
        alpha[i] = lambda_ / (1 + 2 * lambda_ - lambda_ * alpha[i - 1])
        beta[i] = (U[j - 1, i] + lambda_ * beta[i - 1]) / (1 + 2 * lambda_ - lambda_ * alpha[i - 1])

    for i in range(I - 2, 0, -1):
        U[j + 1, i] = alpha[i] * U[j + 1, i + 1] + beta[i]

# Визуализация результатов
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(T, X, U, cmap='magma')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('U')
ax.set_title('неявный')
plt.show()