import numpy as np
import matplotlib.pyplot as plt


def Ux0(x):
    return 0 if x >= 0.5 else 1


h = 0.1
T = 0.001
a, b = 0, 1
c, d = 0, 1
eps = 0.01
p = int(b / h) + 1
q = int(d / T) + 1
U = np.zeros((p, q))

for i in range(p):
    x = h * i
    U[i, 0] = Ux0(x)
for j in range(q - 1):
    for i in range(1, p - 1):
        U[i, j + 1] = (
                U[i, j]
                - T / h * U[i, j] * (U[i, j] - U[i - 1, j])
                - eps ** 2 * T / (2 * h ** 3) * (U[i + 1, j]
                                                 - U[i - 1, j]) * (U[i + 1, j] - 2 * U[i, j] + U[i - 1, j])
        )
    U[p - 1, j + 1] = U[p - 1, j] - T / h * U[p - 1, j] * (U[p - 1, j] - U[p - 2, j])

x = np.linspace(0, b, p)
t = np.linspace(0, d, q)
X, T = np.meshgrid(x, t, indexing='ij')
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("U")
ax.plot_surface(X, T, U, cmap='magma')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def Ux0(x):
    return 0 if x >= 0.5 else 1

h = 0.1
T = 0.001
a, b = 0, 1
c, d = 0, 1
p = int(b / h)
q = int(d / T)
U = np.zeros((p, q))

for i in range(p):
    U[i] = [0]*q

for i in range(0,p):
    x = h*i
    U[i][0]= Ux0(x)

for j in range(0,q - 1):
    for i in range(1, p):
        x = h*i
        t = T*j
        U[i][j+1] = U[i][j]-T/(2*h)*(U[i][j]**2 - U[i-1][j]**2)

x = np.linspace(0, b, p)
t = np.linspace(0, d, q)
X, T = np.meshgrid(x, t, indexing='ij')
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("U")
ax.plot_surface(X, T, U, cmap='magma')
plt.show()


