from __future__ import division, print_function

import numpy as np


""" Define the univariate function prototypes
"""

def _line_generator(kwargs):
    """ Generate a line function

    :param kwargs:
      * `xs` - start point of the prototype
      * `xe` - end point of the prototype
      * `fs` - value of the function at the start point
      * `fprime` - value of the derivative of the function at the start point
    :return: an instance of a line prototype.
    """
    xs = kwargs.get('xs')
    xe = kwargs.get('xe')
    fs = kwargs.get('fs')
    fprime = kwargs.get('fprime')

    def wfun(x):
        f = (x - xs) * fprime + fs
        g = fprime
        return f, g

    wfun.__name__ = 'line_fun'

    return wfun


def _relu_generator(kwargs):
    """ Generates a linear rectifier function

    :param kwargs:
      * `xs` - start point of the prototype
      * `xe` - end point of the prototype
      * `fs` - value of the function at the start point
      * `fprime` - value of the derivative of the function at the start point
    :return: an instance of a linear rectifier prototype
    """
    xs = kwargs.get('xs')
    xe = kwargs.get('xe')
    fs = kwargs.get('fs')
    fprime = kwargs.get('fprime')
    l2 = fprime * fprime

    xm = (xe + xs * np.sqrt(1 + l2)) / (1 + np.sqrt(1 + l2))
    fm = fprime * (xm - xs) + fs

    def wfun(x):
        if x > xm:
            f = fm
            g = 0
        else:
            f = (x - xs) * fprime + fs
            g = fprime
        return f, g

    wfun.__name__ = 'relu_fun'

    return wfun


def _angle_generator(kwargs):
    """ Generate angle prototype function
    :param kwargs:
    :return:
    """
    xs = kwargs.get('xs')
    xe = kwargs.get('xe')
    fs = kwargs.get('fs')
    fprime = kwargs.get('fprime')
    kwargs['angle'] = kwargs.get('angle', 90)
    angle = kwargs['angle']

    ls = fprime
    angle = angle * np.pi / 180

    if kwargs['angle'] == 90:
        le = -1 / ls
    else:
        tanf = np.tan(angle)
        le = (ls - tanf) / (1 + ls * tanf)

    gamma = np.sqrt((1 + ls * ls) / (1 + le * le))
    xm = (gamma * xs + xe) / (1 + gamma)
    fm = ls * (xm - xs) + fs
    # fe = le * (xe - xm) + fm  # TODO: unused ?

    def wfun(x):
        if x > xm:
            f = (x - xm) * le + fm
            g = le
        elif x < xm:
            f = (x - xs) * ls + fs
            g = ls
        else:
            f = fm
            g = 0
        return f, g

    wfun.__name__ = 'angle_fun'

    return wfun


def _abs_generator(kwargs):
    """ Generate absolute value function
    :param kwargs:
    :return:
    """
    fprime = kwargs.get('fprime')
    thetas = np.arctan(fprime) * 180 / np.pi
    kwargs['angle'] = 2 * thetas - 180

    wfun = _angle_generator(kwargs)
    wfun.__name__ = 'abs_fun'

    return wfun


def _bend_generator(kwargs):
    default_angle = -45 * np.pi / 180

    sign = kwargs.get('sign', 1)
    fprime_e = kwargs.get('oldfprime') or kwargs.get('fprime') or \
        np.tan(default_angle)
    fprime_e = fprime_e * sign

    kwargs['angle'] = 180 - np.arctan(fprime_e) * 180 / np.pi
    kwargs['fprime'] = 0

    wfun = _angle_generator(kwargs)
    wfun.__name__ = 'bend_fun'

    return wfun


def _cliff_generator(kwargs):
    assert kwargs['fprime'] != 0, \
        "Error the gradient before the cliff cannot be 0"

    fprime = kwargs['fprime']
    sign = kwargs.get('sign', -1)
    steepness = sign * 10

    ls = fprime
    le = fprime * steepness
    theta_s = np.arctan(ls)
    theta_e = np.arctan(le)
    angle = theta_s - theta_e
    angle = angle * 180 / np.pi

    kwargs['angle'] = angle

    wfun = _angle_generator(kwargs)
    wfun.__name__ = 'cliff_fun'

    return wfun


def _ridge_generator(opt):
    assert opt['fprime'] != 0, \
        "Error the gradient before the ridge cannot be 0"

    fprime = opt['fprime']
    sign = opt.get('sign', 1)
    steepness = sign * 10

    ls = fprime
    le = fprime * steepness
    theta_s = np.arctan(ls)
    theta_e = np.arctan(le)
    angle = theta_s - theta_e
    angle = angle * 180 / np.pi
    opt['angle'] = angle

    wfun = _angle_generator(opt)
    wfun.__name__ = 'ridge_fun'

    return wfun


def _expon_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']

    b = np.log(1000) / (xe - xs)
    a = -fprime * np.exp(b * xs) / b
    c = fs - a * np.exp(-b * xs)

    def wfun(x):
        f = np.exp(-b * x)
        g = -a * b * f
        f = a * f + c
        return f, g

    wfun.__name__ = 'expon_fun'

    return wfun


def _quad_bowl_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']

    # Assumption fs == fe
    xo = (xs + xe) / 2
    a = fprime / (2 * (xs - xo))
    c = fs - a * (xs - xo) * (xs - xo)

    def wfun(x):
        f = (x - xo)
        g = 2 * a * f
        f = f * f * a + c
        return f, g

    wfun.__name__ = 'quad_bowl'

    return wfun


def _quad_convex_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']

    # Assumption the derivative of f at e is 1/10 of the derivative of f at s
    #  if fprime negative or 10 times more if fprime positive

    if fprime > 0:
        xo = (10 * xs - xe) / 9
    else:
        xo = (xe - 0.1 * xs) / 0.9

    a = fprime / (2 * (xs - xo))
    c = fs - a * (xs - xo) * (xs - xo)

    def wfun(x):
        f = (x - xo)
        g = 2 * a * f
        f = f * f * a + c
        return f, g

    wfun.__name__ = 'quad_convex'

    return wfun


def _gauss_bowl_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']
    if fprime == 0:
        fprime = -1e-7

    # Assumption in the bowl case: the xs=mu-0.1*c, xe=mu+0.1*c
    c = 5 * (xe - xs)
    mu = xs + 0.1 * c

    expon = np.exp(-(xs - mu) * (xs - mu) / (c * c))
    b = (c * c) * fprime / (2 * (xs - mu) * expon)
    a = fs + b * expon

    def wfun(x):
        g = 2 * b * ((x - mu) / (c * c))
        f = np.exp((x - mu) * (x - mu) / (-c * c))
        g = g * f
        f = a - b * f

        return f, g

    return wfun


def _gauss_convex_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']
    if fprime == 0:
        fprime = -1e-7

    # Assumption in the convex case:
    #  from the point where the second derivative is 0
    #  to the point that is 0.1*c away from the mean
    c = 10 * (xe - xs) / (5 * np.sqrt(2) - 1)

    # Descent case
    if fprime > 0:
        mu = xs - 0.1 * c
    else:
        mu = xe + 0.1 * c

    expon = np.exp(-(xs - mu) * (xs - mu) / (c * c))
    b = (c * c) * fprime / (2 * (xs - mu) * expon)
    a = fs + b * expon

    def wfun(x):
        g = 2 * b * ((x - mu) / (c * c))
        f = np.exp((x - mu) * (x - mu) / (-c * c))
        g = g * f
        f = a - b * f
        return f, g

    return wfun


def _gauss_nonconvex_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']
    if fprime == 0:
        fprime = -1e-7

    # Assumption in the non convex case:
    #  the xs point is 4c points away of the mean, where c is the bandwidth
    c = 2 * (xe - xs) / (8 - np.sqrt(2))

    # Descent case
    if fprime > 0:
        mu = xe - 4 * c
    else:
        mu = xs + 4 * c

    expon = np.exp(-(xs - mu) * (xs - mu) / (c * c))
    b = (c * c) * fprime / (2 * (xs - mu) * expon)
    a = fs + b * expon

    def wfun(x):
        g = 2 * b * ((x - mu) / (c * c))
        f = np.exp((x - mu) * (x - mu) / (-c * c))
        g = g * f
        f = a - b * f
        return f, g

    return wfun


def _laplace_bowl_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']

    if fprime == 0:
        fprime = -1e-7

    # Assumption in the bowl case the xs=mu-0.1*c, xe=mu+0.1*c
    c = (xe - xs) / 0.2
    mu = xs + 0.1 * c
    expon = np.exp(-np.abs(xs - mu) / c)
    b = c * fprime / (expon * np.sign(xs - mu))
    a = fs + b * expon

    def wfun(x):
        g = np.sign(x - mu) * (b / c)
        f = np.exp(np.abs(x - mu) / (-c))
        g = g * f
        f = a - b * f
        return f, g

    return wfun


def _laplace_convex_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']
    if fprime == 0:
        fprime = -1e-7

    # Assumption in the non convex case:
    #  if fprime<=0: xs =mu-c, xe=mu-0.1*c else xs=mu+0.1*c, xe=mu+c

    c = (xe - xs) / (1 - 0.1)

    # Ascend case
    if fprime > 0:
        mu = xs - 0.1 * c
    else:
        mu = xs + 1 * c

    expon = np.exp(-np.abs(xs - mu) / c)
    b = c * fprime / (expon * np.sign(xs - mu))
    a = fs + b * expon

    def wfun(x):
        g = np.sign(x - mu) * (b / c)
        f = np.exp(np.abs(x - mu) / (-c))
        g = g * f
        f = a - b * f
        return f, g

    return wfun


def _laplace_nonconvex_generator(opt):
    xs = opt['xs']
    xe = opt['xe']
    fs = opt['fs']
    fprime = opt['fprime']

    if fprime == 0:
        fprime = -1e-7

    # Assumption in the non convex case:
    # if fprime<=0 then xs =mu-8*c, xe=mu-c else xs=mu+c, xe=mu+8*c

    c = (xe - xs) / (8 - 1)

    # Ascend case
    if fprime > 0:
        mu = xs - c
    else:
        mu = xs + 8 * c

    expon = np.exp(-np.abs(xs - mu) / c)
    b = c * fprime / (expon * np.sign(xs - mu))
    a = fs + b * expon

    def wfun(x):
        g = np.sign(x - mu) * (b / c)
        f = np.exp(np.abs(x - mu) / (-c))
        g = g * f
        f = a - b * f
        return f, g

    return wfun


def _offset(fun, xo):
    """This function generates a new function which is an
    offset version of the original function fun by xo.
    """

    def wfun(x):
        return fun(x - xo)

    wfun.__name__ = 'offset_fun'

    return wfun


FUNCTION_PROTOTYPES = {
    'line': _line_generator,
    'relu': _relu_generator,
    'abs': _abs_generator,
    'bend': _bend_generator,
    'cliff': _cliff_generator,
    'ridge': _ridge_generator,
    'expon': _expon_generator,
    'quad_bowl': _quad_bowl_generator,
    'quad_convex': _quad_convex_generator,
    'gauss_bowl': _gauss_bowl_generator,
    'gauss_convex': _gauss_convex_generator,
    'gauss_nonconvex': _gauss_nonconvex_generator,
    'laplace_bowl': _laplace_bowl_generator,
    'laplace_convex': _laplace_convex_generator,
    'laplace_nonconvex': _laplace_nonconvex_generator,
}

SIGN_PROTOTYPES = {
    'bend': True,
    'cliff': True,
    'ridge': True,
}


def make_function(opts, function_names, ascend_descent):
    """
    :param opts:
        * `xs` - start point of the prototype
        * `xe` - end point of the prototype
        * `fs` - value of the function at the start point
        * `fprime` - value of the derivative of the function at the start point
    :return: an instance of a line prototype.
    :param function_names:
        list of function names to be concatenated
    :param ascend_descent:
        list that defines for every function if it is ascended or descended
        This option makes sense only for prototypes with discontinuity
        in their derivative,
        for example in the case of a bend this option defines whether after
        the non differentiable
        point the derivative will have the same or opposite sign as before
        this point.
    :return: a function consisting of the concatenated functions
    """
    if not isinstance(function_names, list):
        function_names = [function_names]
    if not isinstance(ascend_descent, list):
        ascend_descent = [ascend_descent]

    signs = {}
    index_ascend_descend = 0
    for i, function_name in enumerate(function_names):
        if SIGN_PROTOTYPES.get(function_name) is not None:
            signs[i] = ascend_descent[index_ascend_descend]
            index_ascend_descend += 1
        else:
            signs[i] = None

    generated_functions = []
    limits = []

    # Partition the input space [xs, xe] equally over all
    # given functions
    fs = opts['fs']
    xs = opts['xs']
    xe = opts['xe']
    assert xs < xe, "Start limit cannot be smaller than end limit"

    fprime = opts['fprime']
    oldfprime = None
    dx = (xe - xs) / len(function_names)
    xe = xs + dx
    for i, function_name in enumerate(function_names):
        function = FUNCTION_PROTOTYPES[function_name]

        function_opts = {
            'fs': fs,
            'xs': xs,
            'xe': xe,
            'fprime': fprime,
            'oldfprime': oldfprime,
        }
        if i in signs:
            if fprime > 0:
                sign = -1
            elif fprime < 0:
                sign = 1
            elif oldfprime > 0:
                sign = -1
            elif oldfprime < 0:
                sign = 1
            else:
                assert False, "unable to asses sign"
            if signs[i] == 'a':
                sign *= -1
            function_opts['sign'] = sign

        # print("From {0} to {1} use {2}".format(xs, xe, function_name))
        fun = function(function_opts)
        generated_functions.append(fun)
        limits.append(xs)
        if fprime:
            oldfprime = fprime
        fs, fprime = fun(xe)
        xs = xe
        xe = xs + dx
    limits.append(xe)

    xs = opts['xs']
    xe = opts['xe']

    # Create the wrapper function that uses the correct
    # generated function for the input based on the limits
    def wfun(x):
        if x < xs:
            assert False, "x < xs"
        if x > xe:
            assert False, "x > xe"

        for i in xrange(0, len(limits) - 1):
            if limits[i] <= x < limits[i + 1]:
                return generated_functions[i](x)

        raise ValueError("Failed to find a generated function")

    return wfun