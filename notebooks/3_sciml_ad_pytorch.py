import torch
from torch.func import grad
from sympy import symbols, diff, cos, sin
import math


def run_numerical_differences():
    h = 0.001
    x = 1.0
    x0 = math.cos(math.sin(x))
    x1 = math.cos(math.sin(x + h))
    dzdx = (x1 - x0) / h
    print(dzdx)  # -0.40264575946546977


def run_symbolic():
    x = symbols("x")
    y = sin(x)
    z = cos(y)
    dzdx = diff(z, x)
    print(dzdx)  # -sin(sin(x))*cos(x)
    print(dzdx.subs(x, 1.0))  # -0.402862443052853


def run():
    # imperative
    x = torch.ones([], requires_grad=True)
    y = torch.sin(x)
    z = torch.cos(y)
    z.backward()
    print(x.grad)  # tensor(-0.4029)

    # functional
    def h(x):
        return torch.cos(torch.sin(x))

    print(grad(h)(x))  # tensor(-0.4029)


if __name__ == "__main__":
    # run()
    run_numerical_differences()
    # run_symbolic()
