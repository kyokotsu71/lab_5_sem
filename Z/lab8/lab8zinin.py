import numpy as np
import matplotlib.pyplot as plt


a, b = 0, 1
c, d = 0, 10
A = 1
N = 100
M = 1000
h = (b - a) / N
tau = (d - c) / M
lambda_ = (A * tau ** 2) / h ** 2


u = np.zeros((M + 1, N + 1))


x = np.linspace(a, b, N + 1)
u[0, :] = x ** 2
u[1, :] = x ** 2 +2 * tau

for j in range(1, M):
    for i in range(1, N):
        u[j+1, i] = 2 * (1 - lambda_) * u[j, i] + lambda_ * (u[j, i+1] + u[j, i-1]) - u[j-1, i]


    u[j+1, 0] = 0
    u[j+1, N] = 1


t = np.linspace(c, d, M + 1)
X, T = np.meshgrid(x, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u, cmap='plasma')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x, t)')
ax.set_title('явная')

plt.show()

import numpy as np
import matplotlib.pyplot as plt


a, b = 0, 1
c, d = 0, 10
A = 1
N = 100
M = 1000
h = (b - a) / N
tau = (d - c) / M
lambda_ = (A * tau / h) ** 2


u = np.zeros((M + 1, N + 1))


for i in range(N + 1):
    x_i = i * h
    u[0, i] = x_i ** 2

for i in range(N + 1):
    x_i = i * h
    u[1, i] = x_i ** 2 +2* tau


for j in range(1, M):
    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)

    for i in range(1, N):
        alpha[i] = lambda_ / (1 + 2 * lambda_ - lambda_ * alpha[i - 1])
        beta[i] = (2 * u[j, i] - u[j - 1, i] + lambda_ * beta[i - 1]) / (1 + 2 * lambda_ - lambda_ * alpha[i - 1])
    u[j + 1, N] = 1
    for i in range(N - 1, 0, -1):
        u[j + 1, i] = alpha[i] * u[j + 1, i + 1] + beta[i]
    u[j + 1, 0] = 0
x = np.linspace(a, b, N + 1)
t = np.linspace(c, d, M + 1)
X, T = np.meshgrid(x, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u, cmap='magma')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x, t)')
ax.set_title('неявная')
plt.show()

