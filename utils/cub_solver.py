# CUBIC ROOT SOLVER

# Date Created   :    24.05.2017
# Created by     :    Shril Kumar [(shril.iitdhn@gmail.com),(github.com/shril)] &
#                     Devojoyti Halder [(devjyoti.itachi@gmail.com),(github.com/devojoyti)]

# Project        :    Classified
# Use Case       :    Instead of using standard numpy.roots() method for finding roots,
#                     we have implemented our own algorithm which is ~10x faster than
#                     in-built method.

# Algorithm Link :    www.1728.org/cubic2.htm

# This script (Cubic Equation Solver) is an independent program for computation of roots of Cubic Polynomials. This script, however,
# has no relation with original project code or calculations. It is to be also made clear that no knowledge of it's original project
# is included or used to device this script. This script is complete freeware developed by above signed users, and may further be
# used or modified for commercial or non-commercial purpose.


# Libraries imported for fast mathematical computations.
import math

import numpy as np
# Main Function takes in the coefficient of the Cubic Polynomial
# as parameters and it returns the roots in form of numpy array.
# Polynomial Structure -> ax^3 + bx^2 + cx + d = 0
import torch


def solve(a, b, c, d):
    print(f'a: {a:.60e}, b: {b:.60e},c: {c:.60e}, d: {d:.60e}')

    if (a == 0 and b == 0):  # Case for handling Liner Equation
        return np.array([(-d * 1.0) / c])  # Returning linear root as numpy array.

    elif (a == 0):  # Case for handling Quadratic Equations

        D = c * c - 4.0 * b * d  # Helper Temporary Variable
        if D >= 0:
            D = math.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
            return np.array([x1, x2])  # Returning Quadratic Roots as numpy array.
        else:
            return np.array([-1])

    f = findF(a, b, c)  # Helper Temporary Variable
    g = findG(a, b, c, d)  # Helper Temporary Variable
    h = findH(g, f)  # Helper Temporary Variable\
    # print('h_old', h)

    # h0 = g * g
    # h1 = f * f * f
    #
    # h00 = h0 / 4.
    # h11 = h1 / 27.
    # h = h00 + h11

    # print(f'h0: {h0}, h1: {h1}, h00: {h00}, h11: {h11}')

    print(f'f: {f:.60e}')
    print(f'g: {g:.60e}')
    print(f'h: {h:.60e}')


    if f == 0 and g == 0 and h == 0:  # All 3 Roots are Real and Equal
        # print("real_equal_3_mask")
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)
        return np.array([x, x, x])  # Returning Equal Roots as numpy array.

    elif h <= 0:  # All 3 roots are Real\
        i = math.sqrt(((g ** 2.0) / 4.0) - h)  # Helper Temporary Variable
        j = i ** (1 / 3.0)  # Helper Temporary Variable
        k = math.acos(-(g / (2 * i)))  # Helper Temporary Variable
        L = j * -1  # Helper Temporary Variable
        M = math.cos(k / 3.0)  # Helper Temporary Variable
        N = math.sqrt(3) * math.sin(k / 3.0)  # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1  # Helper Temporary Variable\

        print(f'i: {i:.60e}')
        print(f'j: {j:.60e}')
        print(f'k: {k:.60e}')
        print(f'L: {L:.60e}')
        print(f'M: {M:.60e}')
        print(f'N: {N:.60e}')
        print(f'P: {P:.60e}')

        x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        y1 = M - N
        y2 = L * y1
        y3 = y2 + P

        print("y1, y2, y3", y1, y2, y3)
        print("x1, x2, x3", x1, x2, x3)

        return np.array([x1, x2, x3])  # Returning Real Roots as numpy array.

    elif h > 0:  # One Real Root and two Complex Roots
        # print("r1c2_mask")
        R = -(g / 2.0) + math.sqrt(h)  # Helper Temporary Variable
        if R >= 0:
            S = R ** (1 / 3.0)  # Helper Temporary Variable
        else:
            S = (-R) ** (1 / 3.0) * -1  # Helper Temporary Variable
        T = -(g / 2.0) - math.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))  # Helper Temporary Variable
        else:
            U = ((-T) ** (1 / 3.0)) * -1  # Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -1
        x3 = -1

        return np.array([x1, x2, x3])  # Returning One Real Root and two Complex Roots as numpy array.


# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a ** 2.0)) + (27.0 * d / a)) / 27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)


def root_mask(roots):
    return torch.logical_and(roots >= 0, roots <= 1)


def solve_linear_pt(c, d):
    roots = torch.ones_like(c) * -1
    x = -d / c

    x_mask = root_mask(x)
    roots[x_mask] = x[x_mask]
    return roots


def solve_quadratic_geq_pt(b, c, D):
    out_roots = torch.ones_like(b) * -1
    D = torch.sqrt(D)
    x1 = (-c + D) / (2.0 * b)
    x2 = (-c - D) / (2.0 * b)

    x1_mask = root_mask(x1)
    x2_mask = root_mask(x2)

    out_roots[x1_mask] = x1[x1_mask]
    out_roots[x2_mask] = x2[x2_mask]

    return out_roots


def solve_quadratic_l_pt(b, c, D):
    out_roots = torch.ones_like(b) * -1
    D = torch.sqrt(-D)
    x1 = (-c + D * 1j) / (2.0 * b)
    x2 = (-c - D * 1j) / (2.0 * b)

    x1_mask = root_mask(x1)
    x2_mask = root_mask(x2)

    out_roots[x1_mask] = x1[x1_mask]
    out_roots[x2_mask] = x2[x2_mask]

    return out_roots


def solve_quadratic_pt(b, c, d):
    out_roots = torch.ones_like(b) * -1
    D = c * c - 4.0 * b * d  # Helper Temporary Variable
    geq_mask = D >= 0
    l_mask = torch.logical_not(geq_mask)

    out_roots[geq_mask] = solve_quadratic_geq_pt(b[geq_mask], c[geq_mask], D[geq_mask])
    # out_roots[l_mask] = solve_quadratic_l_pt(b[l_mask], c[l_mask], D[l_mask])

    return out_roots


def solve_cubic_req_pt(a, d):
    x = torch.ones_like(a) * -1
    out_roots = torch.ones_like(a) * -1
    dda = d / a
    dda_geq_mask = dda >= 0
    dda_l_mask = torch.logical_not(dda_geq_mask)

    x[dda_geq_mask] = dda[dda_geq_mask] ** (1 / 3.0) * -1
    x[dda_l_mask] = (-dda) ** (1 / 3.0)

    x_mask = root_mask(x)
    out_roots[x_mask] = x[x_mask]

    return out_roots


def solve_cubic_real_pt(a, b, g, h):
    out_roots = torch.ones_like(a) * -1
    i = torch.sqrt(((g ** 2.0) / 4.0) - h)  # Helper Temporary Variable
    j = i ** (1 / 3.0)  # Helper Temporary Variable
    k = torch.acos(-(g / (2 * i)))  # Helper Temporary Variable
    L = j * -1  # Helper Temporary Variable
    M = torch.cos(k / 3.0)  # Helper Temporary Variable
    N = math.sqrt(3) * torch.sin(k / 3.0)  # Helper Temporary Variable
    P = (b / (3.0 * a)) * -1  # Helper Temporary Variable

    x1 = 2 * j * torch.cos(k / 3.0) - (b / (3.0 * a))
    x2 = L * (M + N) + P
    x3 = L * (M - N) + P

    # print("x1, x2, x3", x1.item(), x2.item(), x3.item())

    x1_mask = root_mask(x1)
    x2_mask = root_mask(x2)
    x3_mask = root_mask(x3)

    # print('x1_mask', x1_mask)
    # print('x2_mask', x2_mask)
    # print('x3_mask', x3_mask)

    out_roots[x1_mask] = x1[x1_mask]
    out_roots[x2_mask] = x2[x2_mask]
    out_roots[x3_mask] = x3[x3_mask]
    # print('out_roots', out_roots)

    return out_roots


def solve_cubic_r1c2(a, b, g, h):
    R = -(g / 2.0) + torch.sqrt(h)  # Helper Temporary Variable

    rgeq_mask = R >= 0
    rl_mask = torch.logical_not(rgeq_mask)
    S = torch.ones_like(a) * -1
    S[rgeq_mask] = R[rgeq_mask] ** (1 / 3.0)
    S[rl_mask] = (-R[rl_mask]) ** (1 / 3.0) * -1

    T = -(g / 2.0) - torch.sqrt(h)
    tgeq_mask = T >= 0
    tl_mask = torch.logical_not(tgeq_mask)
    U = torch.ones_like(a) * -1
    U[tgeq_mask] = T[tgeq_mask] ** (1 / 3.0)
    U[tl_mask] = (-T[tl_mask]) ** (1 / 3.0) * -1

    out_roots = torch.ones_like(a) * -1
    x1 = (S + U) - (b / (3.0 * a))
    x1_mask = root_mask(x1)
    out_roots[x1_mask] = x1[x1_mask]

    return out_roots


def solve_cubic_pt(a, b, c, d):
    out_roots = torch.ones_like(a) * -1
    f = findF(a, b, c)  # Helper Temporary Variable
    g = findG(a, b, c, d)  # Helper Temporary Variable
    h = findH(g, f)  # Helper Temporary Variable

    # print('f, g, h', f.item(), g.item(), h.item())
    real_equal_3_mask = torch.logical_and(torch.logical_and(f == 0, g == 0), h == 0)
    real_3_mask = torch.logical_and(torch.logical_not(real_equal_3_mask), h <= 0)
    r1c2_mask = h > 0

    # print('real_equal_3_mask', real_equal_3_mask)
    # print('real_3_mask', real_3_mask)
    # print('r1c2_mask', r1c2_mask)

    out_roots[real_equal_3_mask] = solve_cubic_req_pt(a[real_equal_3_mask], d[real_equal_3_mask])
    out_roots[real_3_mask] = solve_cubic_real_pt(a[real_3_mask], b[real_3_mask], g[real_3_mask], h[real_3_mask])
    out_roots[r1c2_mask] = solve_cubic_r1c2(a[r1c2_mask], b[r1c2_mask], g[r1c2_mask], h[r1c2_mask])

    return out_roots


def solve_torch(a, b, c, d):
    out_roots = torch.ones_like(a) * -1
    linear_mask = torch.logical_and(a == 0, b == 0)
    quadratic_mask = torch.logical_and(a == 0, b != 0)
    cubic_mask = a != 0

    # print('linear_mask', linear_mask)
    # print('quadratic_mask', quadratic_mask)
    # print('cubic_mask', cubic_mask)

    out_roots[linear_mask] = solve_linear_pt(c[linear_mask], d[linear_mask])
    out_roots[quadratic_mask] = solve_quadratic_pt(b[quadratic_mask], c[quadratic_mask], d[quadratic_mask])
    out_roots[cubic_mask] = solve_cubic_pt(a[cubic_mask], b[cubic_mask], c[cubic_mask], d[cubic_mask])
    return out_roots


def cbrtf(x):
    return x ** (1. / 3.)


def solve_like_cuda(a, b, c, d):
    EPSILON = 1e-17
    root = -1

    if (a == 0 and b == 0):  # Case for handling Liner Equation
        return -d / c  # Returning linear root as numpy array.

    elif (a == 0):  # Case for handling Quadratic Equations

        D = c * c - 4.0 * b * d  # Helper Temporary Variable
        if D >= 0:
            D = math.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else:
            D = math.sqrt(-D)
            x1 = (-c + D * 1j) / (2.0 * b)
            x2 = (-c - D * 1j) / (2.0 * b)

        if (x1 >= -EPSILON and x1 <= 1 + EPSILON):
            root = x1
        elif (x2 >= -EPSILON and x2 <= 1 + EPSILON):
            root = x2

        return root

    q = ((3 * c) / a - ((b * b) / (a * a))) / 9
    r = (((-(2 * (b * b * b)) / (a * a * a))) +
         ((9 * (b * c)) / (a * a)) -
         ((27 * d) / a)) / 54
    _del = ((r * r)) + ((q * q * q))

    print('q, r, del', q, r, _del);

    if (_del <= 0):
        theta = math.acos((r / math.sqrt(-(q * q * q))))
        sqrtq = 2 * math.sqrt(-q)

        x1 = (sqrtq * math.cos((theta / 3)) - (b / (a * 3)))
        x2 = (sqrtq * math.cos((theta + 2 * 3.1415927) / 3)) - (b / (a * 3))
        x3 = (sqrtq * math.cos((theta + 4 * 3.1415927) / 3)) - (b / (a * 3))

        if (x1 >= -EPSILON and x1 <= 1 + EPSILON):
            root = x1
        elif (x2 >= -EPSILON and x2 <= 1 + EPSILON):
            root = x2
        elif (x3 >= -EPSILON and x3 <= 1 + EPSILON):
            root = x3

    if (_del > 0):
        x1 = ((cbrtf((r + math.sqrt(_del))))
              + cbrtf((r - math.sqrt(_del)))) - (b / (3 * a))
        if (x1 >= -EPSILON and x1 <= 1 + EPSILON):
            root = x1

    if (q == 0 and r == 0):
        x1 = -(b / 3)
        if (x1 >= -EPSILON and x1 <= 1 + EPSILON):
            root = x1

    return root
