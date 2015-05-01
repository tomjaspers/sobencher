from __future__ import division, print_function

import main.factories.prototypes_functions as prototype_functions
from main.factories.prototypes_noise import wrap_noise
import main.definitions as definitions

import numpy as np
import matplotlib.pyplot as plt

det_func = prototype_functions.make_function(
    definitions.scales['normal'],
    # ['laplace_nonconvex', 'laplace_convex']*1,
    ['quad_bowl']*10,
    # ['line', 'gauss_bowl', 'quad_bowl', 'cliff'],
    ['a'])

func = wrap_noise('gauss', 'add', det_func, scale=0.1)


def plot(f):
    xs = np.linspace(0.1, 0.9, 100)
    plt.scatter(xs, [f_x for (f_x, _) in map(f, xs)], color='gray')
    plt.scatter(xs, [f_x for (f_x, _) in map(f, xs)], color='gray')
    plt.scatter(xs, [f_x for (f_x, _) in map(f, xs)], color='gray')
    plt.scatter(xs, [f_x for (f_x, _) in map(f, xs)], color='gray')
    plt.scatter(xs, [f_x for (f_x, _) in map(f, xs)], color='gray')
    plt.show()


if __name__ == '__main__':
    plot(func)