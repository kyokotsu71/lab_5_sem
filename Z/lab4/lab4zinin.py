import numpy as np
import sympy as sp
import math


def f(x):
    return np.log10(x)


# вычисление многочлена
def lagrange(x, xk, yk):
    n = len(xk)
    L = np.zeros_like(x)
    for i in range(n):
        l = np.ones_like(x)
        for j in range(n):
            if i != j:
                l *= (x - xk[j]) / (xk[i] - xk[j])
        L += l * yk[i]
    return L


def error_lagrange(x, n, tx):
    X = sp.symbols('x')
    derivative = sp.diff(sp.log(X), X, n + 1)
    max_derivative = max(abs(derivative.subs(X, tx[i])) for i in range(n))
    counter = 1
    for X in tx:
        counter *= (x - X)
    error = (max_derivative * abs(counter)) / math.factorial(n + 1)
    return error


xk = np.array([8.1, 8.5, 8.9, 9.3])
yk = f(xk)
x = 8.4
y_exact = f(x)
y_interp = lagrange(np.array([x]), xk, yk)[0]

# погрешность
n = len(xk) - 1
tx = xk
error = error_lagrange(x, n, tx)
absolute_error = abs(y_exact - y_interp)

# вывод
print(f"значение функции аналитически в точке {x}: {y_exact:.8f}")
print(f"значение с помощью многочлена Лагранжа в точке {x}: {y_interp:.8f}")
print(f"погрешность по формуле Лагранжа: {error:.8f}")
print(f"абсолютная погрешность: {absolute_error:.8f}")


import numpy as np

def f(x):
    return 3 ** (x / 2)

# границы
a = 5.4
b = 6.0
h = (b - a) / 4

x_values = np.linspace(a, b, 5)

out = []
n = len(x_values)

for i in range(n):
    xi = x_values[i]
    exact_first = 1/2 * 3**(xi/2) *np.log(3)
    exact_second = 1/4 * 3**(xi/2)*(np.log(3)**2)

    if i == 0 or i == n - 1:
        second = None
    else:
        second = (f(xi + h) - 2 * f(xi) + f(xi - h)) / h ** 2

    if i == 0:
        l = None
        c = None
        r = (f(xi + h) - f(xi)) / h

    elif i == n - 1:
        l = (f(xi) - f(xi - h)) / h
        c = None
        r = None

    else:
        l = (f(xi) - f(xi - h)) / h  # Левая производная
        r = (f(xi + h) - f(xi)) / h  # Правая производная
        c = (f(xi + h) - f(xi - h)) / (2 * h)  # Центральная производная

    error_left = abs(l - exact_first) if l is not None else None
    error_central = abs(c - exact_first) if c is not None else None
    error_right = abs(r - exact_first) if r is not None else None
    error_second = abs(second - exact_second) if second is not None else None

    out.append({
        'x': round(xi, 4),
        'f(x)': round(f(xi), 4),
        'left': round(l, 4) if l is not None else None,
        'right': round(r, 4) if r is not None else None,
        'central': round(c, 4) if c is not None else None,
        'f"(x)': round(second, 4) if second is not None else None,
        'exact 1f dec': round(exact_first, 4),
        'exact 2f dec': round(exact_second, 4),
        'погрешность слева ': round(error_left, 4) if error_left is not None else None,
        'погрешность производной центр': round(error_central, 4) if error_central is not None else None,
        'погрешность производной справа': round(error_right, 4) if error_right is not None else None,
        'погрешность второй производной': round(error_second, 4) if error_second is not None else None
    })

for i in out:
    print(i)