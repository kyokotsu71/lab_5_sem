import numpy as np
import matplotlib.pyplot as plt


def U_x(x):
    return x ** 2 - 25


def U_1(t):
    return t ** 2 - 25


def U_2(t):
    return t ** 2 - 24


def f(x, t):
    return 2 * x


def scheme1(i, j, f, x, tau, U, t):
    for k in range(1, i + 1):
        for l in range(0, j):
            U[k][l + 1] = U[k - 1][l] + tau * f(x[k], t[l])
    return U


def scheme2(i, j, f, x, tau, U, t):
    for k in range(i - 1, -1, -1):
        for l in range(0, j):
            U[k][l + 1] = U[k + 1][l] - tau * f(x[k], t[l])
    return U


def scheme3(i, j, f, x, U, tau, a, t):
    if a <= 0:
        for k in range(i - 1, -1, -1):
            for l in range(0, j):
                U[k][l + 1] = (U[k][l] + U[k + 1][l + 1] - tau * f(x[k], t[l])) / 2
    else:
        for k in range(1, i + 1):
            for l in range(0, j):
                U[k][l + 1] = (U[k][l] + U[k - 1][l + 1] + tau * f(x[k], t[l])) / 2
    return U


def scheme4(i, j, f, x, U, tau, a, h, t):
    if a <= 0:
        for k in range(i - 1, -1, -1):
            for l in range(0, j):
                U[k][l + 1] = (U[k + 1][l] - tau * f(x[k] + h / 2, t[l] + tau / 2))
    else:
        for k in range(1, i + 1):
            for l in range(0, j):
                U[k][l + 1] = (U[k - 1][l] + tau * f(x[k] + h / 2, t[l] + tau / 2))
    return U


def init(I, J, dx, dt, a, U_x, U_t, rect):
    if rect:
        I1 = I
    else:
        I1 = I + J
    U = np.zeros((I1 + 1, J + 1))
    x = np.zeros(I1 + 1)
    t = np.zeros(J + 1)
    if rect:
        for i in range(I + 1):
            x[i] = dx * i
            U[i][0] = U_x(x[i])
        if a > 0:
            for j in range(J + 1):
                t[j] = dt * j
                if j != 0:
                    U[0][j] = U_t(t[j])
        elif a < 0:
            for j in range(J + 1):
                t[j] = dt * j
                if j != 0:
                    U[I][j] = U_t(t[j])
    else:
        if a > 0:
            for i in range(I + J + 1):
                x[i] = i * dx
                U[i][0] = U_x(x[i])
        elif a < 0:
            for i in range(I + J + 1):
                x[i] = i * dx
                U[i][0] = U_x(x[i])
                for j in range(J + 1):
                    t[j] = j * tau
    return x, t, U


def draw_plot(graph):
    graph[0], graph[1] = np.meshgrid(graph[0], graph[1])
    graph[0], graph[1] = graph[0].T, graph[1].T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("U")
    ax.set_rasterization_zorder(1)
    ax.plot_surface(graph[0], graph[1], graph[2], cmap='viridis')
    plt.show()



x_start = 0
x_end = 1
time_start = 0
time_end = 10
h = 0.1
a1 = 2
a2 = -2
tau = h / max([abs(a1), abs(a2)])

I = int((x_end - x_start) / h)
J = int((time_end - time_start) / tau)
graph = []
print('Схема 1 а>0')

x, t, U = init(I, J, h, tau, a1, U_x, U_1, False)
U = scheme1(I + J, J, f, x, tau, U, t)
x, t = np.linspace(x_start, x_end, I + 1), np.linspace(time_start, time_end, J + 1)
U=U[J:I+J+1, :]
draw_plot([x, t, U])




print('Схема 2 а<0')
x, t, U = init(I, J, h, tau, a2, U_x, U_1, False)
U = scheme2(I + J, J, f, x, tau, U, t)
x, t = np.linspace(x_start, x_end, I + 1), np.linspace(time_start, time_end, J + 1)
U = U[0:I + 1, :]
draw_plot([x, t, U])

print('Схема 3 а>0')
x, t, U = init(I, J, h, tau, a1, U_x, U_1, True)
U = scheme1(I, J, f, x, tau, U, t)
x, t = np.linspace(x_start, x_end, I + 1), np.linspace(time_start, time_end, J + 1)
draw_plot([x, t, U])


print('Схема 4 а>0')
x, t, U = init(I, J, h, tau, a1, U_x, U_1, True)
U = scheme3(I, J, f, x, U, tau, a1, t)
x, t = np.linspace(x_start, x_end, I + 1), np.linspace(time_start, time_end, J + 1)
draw_plot([x, t, U])


print('Схема 5 а>0')
x, t, U = init(I, J, h, tau, a1, U_x, U_2, True)
U = scheme4(I, J, f, x, U, tau, a1, h, t)
x, t = np.linspace(x_start, x_end, I + 1), np.linspace(time_start, time_end, J + 1)
draw_plot([x, t, U])


print('Схема 6 а<0')
x, t, U = init(I, J, h, tau, a2, U_x, U_2, True)
U = scheme2(I, J, f, x, tau, U, t)
x, t = np.linspace(x_start, x_end, I + 1), np.linspace(time_start, time_end, J + 1)
draw_plot([x, t, U])


print('Схема 7 а<0')
x, t, U = init(I, J, h, tau, a2, U_x, U_2, True)
U = scheme4(I, J, f, x, U, tau, a2, h, t)
x, t = np.linspace(x_start, x_end, I + 1), np.linspace(time_start, time_end, J + 1)
draw_plot([x, t, U])

