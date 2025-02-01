import math
import sympy as sp

def f(x):
    return (math.exp(x) - 1) / (math.exp(x) + 1)

def F(x):
    return 2 * math.log(math.exp(x) + 1) - x

def left_rectangles(f, a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(n):
        sum += f(a + i * h)
    return h * sum

def right_rectangles(f, a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(1, n + 1):
        sum += f(a + i * h)
    return h * sum

def center_rectangles(f, a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(1, n + 1):
        sum += f(a + (i - 0.5) * h)  # Center of the interval
    return h * sum

def simpson(f, a, b, n):
    h = (b - a) / n
    k = 0
    x = a + h
    for i in range(1, n // 2 + 1):
        k += 4 * f(x)
        x += 2 * h

    x = a + 2 * h
    for i in range(1, n // 2):
        k += 2 * f(x)
        x += 2 * h
    return (h / 3) * (f(a) + f(b) + k)


def trap(f, a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(1, n):
        sum += f(a + i * h)
    return h * ((f(a)+f(b)) / 2 + sum)

x = sp.symbols('x')
function = (sp.exp(x) - 1) / (sp.exp(x) + 1)
integral_value = sp.integrate(function, (x, 0, 3))
exact = integral_value.evalf()
print(exact)
print()

results_tab = {}
acc = 10 ** (-4)
a, b = 0, 3

for method in [left_rectangles, right_rectangles, center_rectangles, trap, simpson]:
    n = 2
    result = method(f, a, b, n)
    prev_result = result
    while True:
        n *= 2
        result = method(f, a, b, n)
        relative_error = abs((exact - result) / exact) * 100
        if abs(prev_result - result) < acc:
            break
        prev_result = result

    results_tab[method.__name__] = {
        'value': result,
        'step_size': (b - a) / n,
        'n': n,
        'relative_error': relative_error
    }

print(f"{'Метод':<30}{'Значение интеграла':<25}{'Шаг':<25}{'Количество разбиений':<30}{'Относительная погрешность (%)'}")
for method, res in results_tab.items():
    print(f"{method:<30}{round(res['value'], 4):<25}{round(res['step_size'], 4):<25}{res['n']:<30}{res['relative_error']:.4f}")
