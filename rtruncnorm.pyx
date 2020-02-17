%%cython --compile-args=-fopenmp --link-args=-fopenmp --force

from libc.float cimport DBL_MIN
from libc.math cimport INFINITY, NAN, exp, fabs, log, sqrt
from libc.stdlib cimport rand, RAND_MAX

cimport cython
cimport openmp
from cython.parallel cimport prange

cimport numpy as np
import numpy as np


cdef double M_1_SQRT_2PI = 0.398942280401432677939946059934
cdef double M_LN_SQRT_2PI = 0.918938533204672741780329736406
cdef double log_t1 = log(0.15)
cdef double log_t2 = log(2.18)
cdef double t3 = 0.725
cdef double t4 = 0.45


# Generate a U(a,b) random variable.
@cython.cdivision(True)
cdef inline double runif(double a, double b) nogil:
    cdef double u = rand()
    u /= RAND_MAX
    return a + (b - a) * u


# Generate an Exp(rate) random variable.
@cython.cdivision(True)
cdef inline double rexp(double rate) nogil:
    cdef double u = runif(0.0, 1.0)
    return -log(u) / rate


# Generate an N(mean, sd) random variable.
@cython.cdivision(True)
cdef inline double rnorm(double mean, double sd) nogil:
    cdef double x1, x2
    cdef double w = 2.0

    while w >= 1.0:
        x1 = 2.0 * runif(0.0, 1.0) - 1.0
        x2 = 2.0 * runif(0.0, 1.0) - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt(-2.0 * log(w) / w)
    return mean + sd * x1 * w


cdef inline double dnorm(double x, int give_log) nogil:
    x = fabs(x)

    if give_log:
        return -(M_LN_SQRT_2PI + 0.5 * x * x)
    else:
        return M_1_SQRT_2PI * exp(-0.5 * x * x)


# Exponential rejection sampling (a,inf).
@cython.cdivision(True)
cdef double ers_a_inf(double a) nogil:
    cdef double x, rho

    x = rexp(a) + a
    rho = exp(-0.5 * (x - a) * (x - a))

    while runif(0.0, 1.0) > rho:
        x = rexp(a) + a
        rho = exp(-0.5 * (x - a) * (x - a))

    return x


# Exponential rejection sampling (a,b).
@cython.cdivision(True)
cdef double ers_a_b(double a, double b) nogil:
    cdef double x, rho

    x = rexp(a) + a
    rho = exp(-0.5 * (x - a) * (x - a))

    while runif(0.0, 1.0) > rho or x > b:
        x = rexp(a) + a
        rho = exp(-0.5 * (x - a) * (x - a))

    return x


# Normal rejection sampling (a,b).
cdef double nrs_a_b(double a, double b) nogil:
    cdef double x = DBL_MIN

    while x < a or x > b:
        x = rnorm(0.0, 1.0)

    return x


# Normal rejection sampling (a,inf).
cdef double nrs_a_inf(double a) nogil:
    cdef double x = DBL_MIN

    while x < a:
        x = rnorm(0.0, 1.0)

    return x


# Half-normal rejection sampling.
cdef double hnrs_a_b(double a, double b) nogil:
    cdef double x = a - 1.0

    while x < a or x > b:
        x = rnorm(0.0, 1.0)
        x = fabs(x)

    return x


# Uniform rejection sampling.
cdef double urs_a_b(double a, double b) nogil:
    cdef double phi_a = dnorm(a, 0)

    # Upper bound of the normal density on [a, b].
    cdef double ub = M_1_SQRT_2PI if a < 0.0 and b > 0.0 else phi_a
    cdef double x = runif(a, b)

    while runif(0.0, 1.0) * ub > dnorm(x, 0):
        x = runif(a, b)

    return x


# Previously, this was referred to as type 1 sampling.
@cython.cdivision(True)
cdef double r_lefttruncnorm(double a, double mean, double sd) nogil:
    cdef double alpha = (a - mean) / sd

    if alpha < t4:
        return mean + sd * nrs_a_inf(alpha)
    else:
        return mean + sd * ers_a_inf(alpha)


@cython.cdivision(True)
cdef double r_righttruncnorm(double b, double mean, double sd) nogil:
    cdef double beta = (b - mean) / sd

    # Exploit symmetry.
    return mean - sd * r_lefttruncnorm(-beta, 0.0, 1.0)


@cython.cdivision(True)
cdef double r_truncnorm(double a, double b, double mean, double sd) nogil:
    cdef double alpha = (a - mean) / sd
    cdef double beta = (b - mean) / sd
    cdef double log_phi_a = dnorm(alpha, 1)
    cdef double log_phi_b = dnorm(beta, 1)

    if beta <= alpha:
        return NAN
    elif alpha <= 0.0 and 0.0 <= beta:  # 2
        if log_phi_a <= log_t1 or log_phi_b <= log_t1:  # 2 (a)
            return mean + sd * nrs_a_b(alpha, beta)
        else:  # 2 (b)
            return mean + sd * urs_a_b(alpha, beta)
    elif alpha > 0.0:  # 3
        if log_phi_a - log_phi_b <= log_t2:  # 3 (a)
            return mean + sd * urs_a_b(alpha, beta)
        else:
            if alpha < t3:  # 3 (b)
                return mean + sd * hnrs_a_b(alpha, beta)
            else:  # 3 (c)
                return mean + sd * ers_a_b(alpha, beta)
    else:  # 3s
        if log_phi_b - log_phi_a <= log_t2:  # 3s (a)
            return mean - sd * urs_a_b(-beta, -alpha)
        else:
            if beta > -t3:  # 3s (b)
                return mean - sd * hnrs_a_b(-beta, -alpha)
            else:  # 3s (c)
                return mean - sd * ers_a_b(-beta, -alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void do_rtruncnorm(int n, double[:] a, double[:] b, double[:] mean, double[:] sd, double[:] ret) nogil:
    cdef double ca, cb, cmean, csd
    cdef int i

    for i in range(n):
        ca = a[i]
        cb = b[i]
        cmean = mean[i]
        csd = sd[i]

        if -INFINITY < ca and cb < INFINITY:
            ret[i] = r_truncnorm(ca, cb, cmean, csd)
        elif -INFINITY == ca and cb < INFINITY:
            ret[i] = r_righttruncnorm(cb, cmean, csd)
        elif -INFINITY < ca and INFINITY == cb:
            ret[i] = r_lefttruncnorm(ca, cmean, csd)
        elif -INFINITY == ca and INFINITY == cb:
            ret[i] = rnorm(cmean, csd)
        else:
            ret[i] = NAN


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef truncnorm_sample(double[:] loc, double[:] scale, double[:] a, double[:] b):
    n = loc.shape[0]
    ret = np.empty(n, dtype="f8", order="C")
    do_rtruncnorm(n, a, b, loc, scale, ret)
    return ret
