from __future__ import division, print_function

import numpy as np

"""
Some noise wrappers. These functions can be used to introduce different types
of noise to the gradient of a deterministic function.
"""


def _gaus_pdf():
    """ Returns the pdf of a gaussian with 0 mean and 1 std
    """
    norm = 1/np.sqrt(2*np.pi)

    def wfun(x):
        return np.exp(-(x*x)/2)/norm

    return wfun


def _cauchy_pdf():
    """ Returns the pdf of a standard cauchy distribution
    """
    norm = 1/np.pi

    def wfun(x):
        return (1/(x*x+1)) * norm

    return wfun


def _additive_noise(scale, fun):
    """ Returns a function that adds noise to the gradient
    of a deterministic function
    """
    def wfun(x, noise):
        f, g = fun(x)
        n = noise * scale
        f = f + x*n
        g = g + n
        return f, g

    return wfun


def _multiplicative_noise(scale, fun):
    """ Returns a function that multiplies the gradient
    of a deterministic function with noise
    """
    def wfun(x, noise):
        f, g = fun(x)
        n = np.exp(noise * scale)
        f = f * n
        g = g * n
        return f, g

    return wfun


NOISE_COMBINERS = {'add': _additive_noise,
                   'mul': _multiplicative_noise, }

NOISE_TYPES = {'gauss': np.random.rand,
               'cauchy': np.random.standard_cauchy}


def wrap_noise(noise_type, noise_combiner, deterministic_fun, scale=0.1):
    if noise_type not in NOISE_TYPES:
        raise ValueError("invalid noise_type")
    if noise_combiner not in NOISE_COMBINERS:
        raise ValueError("invalid noise_combiner")

    fun_add_noise = NOISE_COMBINERS[noise_combiner](scale, deterministic_fun)

    def wfun(x):
        noise = NOISE_TYPES[noise_type]()
        return fun_add_noise(x, noise)

    return wfun
