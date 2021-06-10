import numpy as np

# f: R^k -> R^k is the function for which we want
# to find a root, df is its total derivative.
# x0 or (x0, y0) is our initial guess for the root,
# n is the number of iterations.


def method_onedim(f, df, x0, n):
    for _ in range(n):
        x0 = x0 - f(x0) / df(x0)

    return x0


def method_twodim(f, df, x0, y0, n):

    p = np.array([])
    for _ in range(n):
        p = np.array([x0, y0])
        p = p - np.linalg.inv(df(x0, y0)).dot(f(x0, y0))
        x0 = p[0]
        y0 = p[1]

    return p
