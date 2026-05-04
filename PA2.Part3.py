###############################################################################
# Program: PA2.Part.3.py
# Author(s): Griffin Fee, Jack Zettlemoyer, Kai Behrens, Jacob Fitzmaurice
# Date: 5/3/26
# Purpose: This program answers Part 3 of Math 248, Programming
#          Assignment 2: counting the number of real roots of
#
#              f(x) = ( e^(2x-1) - 2x^2 - 1/2 )
#                   * ( cos(x) - e^(-x^2) + 1 )
#                   * ( x^5 - 9x^4 - x^3 + 17x^2 - 8x - 8 )
#
#          on the interval [-10, 10].
#
#          The strategy exploits the factored structure of f. By the
#          Zero Product Property, the real roots of f are exactly the
#          union of the real roots of its three factors:
#               g(x) = e^(2x-1) - 2x^2 - 1/2
#               h(x) = cos(x) - e^(-x^2) + 1
#               k(x) = x^5 - 9x^4 - x^3 + 17x^2 - 8x - 8
#          We analyze each factor independently using the root-finding
#          methods from Part 1 (especially Bisection and Newton) plus
#          theoretical bounds (FTA,
#          Rolle's theorem, Taylor expansion), then combine the
#          results, taking care not to double-count any x-value that
#          happens to be a root of more than one factor.
#
#          The final answer is 12 real roots on [-10, 10]. Four of
#          these roots (those near +/- 3*pi from h) cannot be isolated
#          in IEEE-754 double precision because they lie at distance
#          ~7e-20 from the floating-point representation of +/- 3*pi,
#          well below the ~1.8e-15 spacing of representable doubles
#          there.  The code therefore counts them by ANALYTIC
#          argument (Taylor expansion of h near +/- 3*pi), which is
#          rigorously justified in the printed report.
#
#          To use, run this file directly from the command line:
#               >> python3 PA2.Part.3.py
#          A complete report of computational evidence and the final
#          root count is printed to stdout.
###############################################################################
# Inputs:  None (the question, function, and interval are fixed by
#          the assignment).
# Outputs: A formatted report printed to stdout containing, for each
#          factor: the theoretical upper bound on the number of roots,
#          the brackets located by a sign-walk, the refined roots
#          obtained by bisection / Newton's method, notes about why Fixed
#          Point Iteration is not useful here, and a tangent-root
#          check at every critical point.  The script ends by printing
#          the total number of real roots of f on [-10, 10].
###############################################################################

import math                                  # for log, exp, cos, pi, sqrt
import numpy as np                           # for vectorized sign-walk arrays


# =============================================================================
# PART 1 METHODS: NEWTON, BISECTION, SECANT
# -----------------------------------------------------------------------------
# Reproduced verbatim from PA2.Part.1.py so this file is self-contained.
# A grader only needs PA2.Part.3.py to reproduce every Part 3 result.
# =============================================================================


def newtons_method(f, f_prime, x0, tol=1e-10, max_iter=100, return_iterates=False):
    """
    Newton's Method for root-finding.

    Computes an approximate root of f(x) = 0 using the iteration:
        x_{n+1} = x_n - f(x_n) / f'(x_n)

    Parameters
    ----------
    f : callable
        Function whose root we seek.
    f_prime : callable
        Derivative of f.
    x0 : float
        Initial guess.
    tol : float, optional
        Convergence tolerance for |x_{n+1} - x_n|. Default 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default 100.
    return_iterates : bool, optional
        If True, also return the full sequence of iterates and
        residuals.

    Returns
    -------
    root : float
        Approximate root.
    num_iterations : int
        Number of iterations performed.
    iterates, residuals : lists
        (Only when return_iterates=True.)

    Raises
    ------
    ZeroDivisionError
        If f'(x_n) = 0 at any iterate.
    """

    # ---- Variable declarations ----
    x_current = float(x0)          # current iterate x_n (double float)
    x_next = 0.0                   # next iterate x_{n+1} (double float)
    f_val = 0.0                    # f(x_n) (double float)
    f_prime_val = 0.0              # f'(x_n) (double float)
    diff = 0.0                     # |x_{n+1} - x_n| (double float)
    iteration_count = 0            # number of iterations (integer)

    if return_iterates:
        iterates_list = [x_current]              # iterate history
        residuals_list = [abs(f(x_current))]     # |f(x_n)| history

    # ---- Main iteration loop ----
    for iteration_count in range(1, max_iter + 1):

        # Step 1: Evaluate f and f' at current iterate
        f_val = f(x_current)
        f_prime_val = f_prime(x_current)

        # Step 2: Guard against zero derivative
        if f_prime_val == 0.0:
            raise ZeroDivisionError(
                f"Newton's method failed: f'(x) = 0 at x = {x_current} "
                f"(iteration {iteration_count})."
            )

        # Step 3: Compute next iterate via Newton's update
        x_next = x_current - f_val / f_prime_val

        # Step 4: Compute step size
        diff = abs(x_current - x_next)

        # Step 5: Record iterate if requested
        if return_iterates:
            iterates_list.append(x_next)
            residuals_list.append(abs(f(x_next)))

        # Step 6: Update current iterate
        x_current = x_next

        # Step 7: Check convergence
        if diff < tol:
            break

    if return_iterates:
        return x_current, iteration_count, iterates_list, residuals_list
    else:
        return x_current, iteration_count


# =============================================================================


def bisection_method(f, a, b, tol=1e-10, max_iter=200, return_iterates=False):
    """
    Bisection Method for root-finding.

    Repeatedly halves an interval [a, b] containing a sign change,
    keeping the half where the sign change persists.

    Parameters
    ----------
    f : callable
        Function whose root we seek.
    a, b : float
        Interval endpoints with f(a)*f(b) < 0.
    tol : float, optional
        Convergence tolerance on (b - a). Default 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default 200.
    return_iterates : bool, optional
        If True, also return the full sequence of midpoints and
        residuals.

    Returns
    -------
    root : float
        Approximate root (final midpoint).
    num_iterations : int
        Number of iterations performed.
    iterates, residuals : lists
        (Only when return_iterates=True.)

    Raises
    ------
    ValueError
        If f(a)*f(b) >= 0 (no sign-change hypothesis).
    """

    # ---- Variable declarations ----
    a = float(a)                   # left endpoint (double float)
    b = float(b)                   # right endpoint (double float)
    m = 0.0                        # midpoint (double float)
    f_a = f(a)                     # f at left endpoint (double float)
    f_b = f(b)                     # f at right endpoint (double float)
    f_m = 0.0                      # f at midpoint (double float)
    iteration_count = 0            # number of iterations (integer)

    # ---- Sign-change hypothesis ----
    if f_a * f_b > 0:
        raise ValueError(
            f"Bisection requires f(a)*f(b) < 0, but f({a})*f({b}) = "
            f"{f_a * f_b:.6e} >= 0."
        )

    if f_a == 0:
        return (a, 0, [a], [0.0]) if return_iterates else (a, 0)
    if f_b == 0:
        return (b, 0, [b], [0.0]) if return_iterates else (b, 0)

    if return_iterates:
        iterates_list = []
        residuals_list = []

    # ---- Main iteration loop ----
    for iteration_count in range(1, max_iter + 1):

        # Step 1: Compute midpoint
        m = (a + b) / 2.0

        # Step 2: Evaluate f at midpoint
        f_m = f(m)

        # Step 3: Record iterate if requested
        if return_iterates:
            iterates_list.append(m)
            residuals_list.append(abs(f_m))

        # Step 4: Exact-zero short-circuit
        if f_m == 0:
            break

        # Step 5: Choose sub-interval that retains sign change
        if f_a * f_m < 0:
            b = m
            f_b = f_m
        else:
            a = m
            f_a = f_m

        # Step 6: Width-based convergence check
        if (b - a) < tol:
            break

    if return_iterates:
        return m, iteration_count, iterates_list, residuals_list
    else:
        return m, iteration_count


# =============================================================================


def secant_method(f, x0, x1, tol=1e-10, max_iter=100, return_iterates=False):
    """
    Secant Method for root-finding.

    Uses the iteration:
        x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    Parameters
    ----------
    f : callable
        Function whose root we seek.
    x0, x1 : float
        Two initial guesses.
    tol : float, optional
        Convergence tolerance for |x_{n+1} - x_n|. Default 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default 100.
    return_iterates : bool, optional
        If True, also return iterate / residual history.

    Returns
    -------
    root : float
        Approximate root.
    num_iterations : int
        Number of iterations.
    iterates, residuals : lists
        (Only when return_iterates=True.)

    Raises
    ------
    ZeroDivisionError
        If f(x_n) = f(x_{n-1}) at any iterate.
    """

    # ---- Variable declarations ----
    x_current = float(x1)          # current iterate x_n (double float)
    x_previous = float(x0)         # previous iterate x_{n-1} (double float)
    x_next = 0.0                   # next iterate x_{n+1} (double float)
    f_val_curr = 0.0               # f(x_n) (double float)
    f_val_prev = 0.0               # f(x_{n-1}) (double float)
    diff = 0.0                     # |x_{n+1} - x_n| (double float)
    iteration_count = 0            # iteration counter (integer)

    if return_iterates:
        iterates_list = [x_previous, x_current]
        residuals_list = [abs(f(x_previous)), abs(f(x_current))]

    # ---- Main iteration loop ----
    for iteration_count in range(1, max_iter + 1):

        # Step 1: Evaluate f at current and previous iterates
        f_val_curr = f(x_current)
        f_val_prev = f(x_previous)

        # Step 2: Guard against horizontal secant line
        if f_val_curr == f_val_prev:
            raise ZeroDivisionError(
                f"Secant failed: f(x_n) = f(x_{{n-1}}) at x = {x_current}."
            )

        # Step 3: Compute next iterate via secant update
        x_next = x_current - f_val_curr * (
            (x_current - x_previous) / (f_val_curr - f_val_prev)
        )

        # Step 4: Compute step size
        diff = abs(x_current - x_next)

        # Step 5: Record iterate if requested
        if return_iterates:
            iterates_list.append(x_next)
            residuals_list.append(abs(f(x_next)))

        # Step 6: Shift previous and current iterates
        x_previous = x_current
        x_current = x_next

        # Step 7: Check convergence
        if diff < tol:
            break

    if return_iterates:
        return x_current, iteration_count, iterates_list, residuals_list
    else:
        return x_current, iteration_count


# =============================================================================
# FIXED POINT ITERATION (Part 3 specific use)
# -----------------------------------------------------------------------------
# Given a continuous map phi: R -> R, FPI iterates x_{n+1} = phi(x_n).
# If phi has a fixed point r and |phi'(r)| < 1, the iteration converges
# locally to r.  We use FPI in this file as an INDEPENDENT cross-check
# on roots found by bisection: if a root r of f(x) = 0 can be rewritten
# as a fixed point r = phi(r) of some contractive phi, then convergence
# of FPI starting near r confirms r is genuinely a root and is not an
# artifact of bisection's sign-walk discretization.
# =============================================================================


def fixed_point_iteration(phi, x0, tol=1.0e-10, max_iter=500,
                          return_iterates=False, return_converged=False):
    """
    Fixed Point Iteration for finding a fixed point r of phi(x).

    Computes the limit of the iteration:
        x_{n+1} = phi(x_n)
    starting from x_0.  By the Banach fixed point theorem, if phi is
    a contraction on a neighborhood containing the fixed point r,
    iterates converge to r linearly with rate |phi'(r)|.

    Parameters
    ----------
    phi : callable
        Iteration map.  A fixed point r satisfies r = phi(r).
    x0 : float
        Initial guess.
    tol : float, optional
        Convergence tolerance for |x_{n+1} - x_n|. Default 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default 500 (FPI may be slow).
    return_iterates : bool, optional
        If True, also return iterate history.
    return_converged : bool, optional
        If True, also return a Boolean indicating whether the method
        stopped because the requested tolerance was reached.  This
        prevents confusing a max-iteration stop with convergence.

    Returns
    -------
    root : float
        Approximate fixed point.
    num_iter : int
        Number of iterations performed.
    iterates : list
        (Only when return_iterates=True.)
    converged : bool
        (Only when return_converged=True.) True if the tolerance was
        reached before the maximum iteration count forced termination.
    """

    # ---- Variable declarations ----
    x_current = float(x0)          # current iterate x_n (double float)
    x_next = 0.0                   # next iterate x_{n+1} (double float)
    diff = 0.0                     # step size |x_{n+1} - x_n| (double float)
    iteration_count = 0            # iteration counter (integer)
    converged = False              # True only if diff < tol occurs

    if return_iterates:
        iterates_list = [x_current]

    # ---- Main iteration loop ----
    for iteration_count in range(1, max_iter + 1):

        # Step 1: Apply phi to obtain the next iterate
        x_next = phi(x_current)

        # Step 2: Compute step size
        diff = abs(x_next - x_current)

        # Step 3: Record iterate if requested
        if return_iterates:
            iterates_list.append(x_next)

        # Step 4: Update current iterate
        x_current = x_next

        # Step 5: Check convergence
        if diff < tol:
            converged = True
            break

    if return_iterates and return_converged:
        return x_current, iteration_count, iterates_list, converged
    if return_iterates:
        return x_current, iteration_count, iterates_list
    if return_converged:
        return x_current, iteration_count, converged
    return x_current, iteration_count


# =============================================================================
# SIGN-WALK UTILITY
# -----------------------------------------------------------------------------
# Given a continuous function and an interval, this helper samples the
# function at evenly spaced grid points and returns the list of small
# sub-intervals where consecutive samples have opposite signs.  By the
# Intermediate Value Theorem each such "bracket" contains at least one
# root.  The grid spacing h controls the resolution; smaller h reduces
# the chance of missing closely-spaced roots, at the cost of more
# function evaluations.
#
# CAVEAT: Sign-walks only detect roots where the function CHANGES sign.
#         An odd-multiplicity root produces a sign change; an
#         even-multiplicity (tangent) root does not.  The per-factor
#         analysis routines compensate for this by separately checking
#         critical points where a tangent root could hide.
# =============================================================================


def sign_change_brackets(func, a, b, h=1e-3):
    """
    Locate every sub-interval [x_i, x_{i+1}] in [a, b] on which the
    continuous function `func` changes sign.

    Parameters
    ----------
    func : callable
        Continuous function R -> R.
    a, b : float
        Interval endpoints with a < b.
    h : float, optional
        Grid spacing. Default 1e-3, giving 20001 samples on [-10, 10].

    Returns
    -------
    brackets : list of (float, float)
        List of brackets (x_i, x_{i+1}) such that
        func(x_i) * func(x_{i+1}) < 0.  By IVT, each bracket contains
        at least one root.  Brackets are returned in left-to-right
        order.

    Notes
    -----
    Samples where func is exactly zero are skipped when comparing
    signs, so an exact root landing on the grid is not double-counted.
    The bracket list is the input expected by `bisection_method`.
    """

    # ---- Variable declarations ----
    n_samples = int(round((b - a) / h)) + 1   # number of grid points (int)
    xs = np.linspace(a, b, n_samples)         # sample x-values (1D array)
    ys = np.array([func(x) for x in xs])      # corresponding y-values (1D array)
    brackets = []                             # output list (list of tuples)
    prev_sign = 0.0                           # last nonzero sign seen (float)
    prev_index = 0                            # index of that sample (int)

    # Initialize prev_sign with the first nonzero sample.  If the very
    # first sample is exactly zero, we skip ahead until we find a
    # sample whose sign is well-defined.
    start = 0
    while start < len(ys) and ys[start] == 0.0:
        start += 1
    if start >= len(ys):
        # Function was identically zero on the entire grid -- no
        # brackets in the usual IVT sense, just return empty.
        return brackets
    prev_sign = np.sign(ys[start])
    prev_index = start

    # Walk forward through the grid and record every sign change.
    for i in range(start + 1, len(ys)):
        s = np.sign(ys[i])
        if s == 0.0:
            # Exact zero on the grid; do not record a "sign change"
            # here -- the caller can detect this separately.  We do
            # NOT update prev_sign so that the bracket spanning this
            # zero is still detectable.
            continue
        if s != prev_sign:
            # Sign change detected between sample prev_index and i.
            # The bracket is [xs[prev_index], xs[i]].
            brackets.append((float(xs[prev_index]), float(xs[i])))
        prev_sign = s
        prev_index = i

    return brackets


# =============================================================================
# SECTION A: THE FUNCTIONS f, g, h, k AND THEIR DERIVATIVES
# -----------------------------------------------------------------------------
# The product f(x) = g(x) * h(x) * k(x), where:
#     g(x) = e^(2x-1) - 2x^2 - 1/2
#     h(x) = cos(x) - e^(-x^2) + 1
#     k(x) = x^5 - 9x^4 - x^3 + 17x^2 - 8x - 8
# Derivatives are needed for Newton's method and for the critical-point
# checks that rule out tangent roots.
# =============================================================================


def k(x):
    """k(x) = x^5 - 9x^4 - x^3 + 17x^2 - 8x - 8 (the polynomial factor)."""
    return x**5 - 9.0*x**4 - x**3 + 17.0*x**2 - 8.0*x - 8.0


def k_prime(x):
    """k'(x) = 5x^4 - 36x^3 - 3x^2 + 34x - 8."""
    return 5.0*x**4 - 36.0*x**3 - 3.0*x**2 + 34.0*x - 8.0


def g(x):
    """g(x) = e^(2x-1) - 2x^2 - 1/2 (the exponential-quadratic factor)."""
    return math.exp(2.0*x - 1.0) - 2.0*x**2 - 0.5


def g_prime(x):
    """g'(x) = 2 e^(2x-1) - 4x."""
    return 2.0*math.exp(2.0*x - 1.0) - 4.0*x


def g_double_prime(x):
    """g''(x) = 4 e^(2x-1) - 4."""
    return 4.0*math.exp(2.0*x - 1.0) - 4.0


def h(x):
    """h(x) = cos(x) - e^(-x^2) + 1 (the trig + Gaussian factor)."""
    return math.cos(x) - math.exp(-x**2) + 1.0


def h_prime(x):
    """h'(x) = -sin(x) + 2x e^(-x^2)."""
    return -math.sin(x) + 2.0*x*math.exp(-x**2)


def f(x):
    """f(x) = g(x) * h(x) * k(x) (the full product)."""
    return g(x) * h(x) * k(x)


# =============================================================================
# SECTION B: HELPER -- REFINE A BRACKET TO A ROOT VIA BISECTION
# =============================================================================


def refine_root_bisection(func, bracket, tol=1e-12):
    """
    Refine a single sign-change bracket to a high-precision root using
    bisection.  Used by every analyze_* routine.

    Parameters
    ----------
    func : callable
        Function whose root we seek.
    bracket : tuple of two floats
        (a, b) with func(a)*func(b) < 0.
    tol : float, optional
        Tolerance for bisection's interval-width stopping criterion.
        Default 1e-12 -- close to double-precision machine epsilon
        scaled by the magnitude of x in [-10, 10].

    Returns
    -------
    root : float
        Refined root.
    num_iter : int
        Number of bisection iterations used.
    """

    # ---- Variable declarations ----
    a = float(bracket[0])           # left endpoint of bracket (double float)
    b = float(bracket[1])           # right endpoint of bracket (double float)
    root = 0.0                      # refined root (double float)
    num_iter = 0                    # iterations used (integer)

    root, num_iter = bisection_method(func, a, b, tol=tol, max_iter=200)
    return root, num_iter


# =============================================================================
# SECTION C: PER-FACTOR ANALYSIS ROUTINES
# -----------------------------------------------------------------------------
# Each analyze_* routine performs a complete, rigorous count of the
# real roots of one factor on [-10, 10], printing all computational
# evidence as it goes.  The routines return the list of refined roots
# they found so that the main script can combine them and check for
# overlaps between factors.
# =============================================================================


def analyze_k():
    """
    Count the real roots of k(x) = x^5 - 9x^4 - x^3 + 17x^2 - 8x - 8
    on [-10, 10].

    Strategy:
      (1) Theoretical upper bound from the Fundamental Theorem of Algebra:
          k has degree 5 -> at most 5 real roots anywhere on R.
      (2) Sign-walk on a fine grid to locate all sign-change brackets.
          By IVT, each bracket contains at least one root of odd
          multiplicity.
      (3) Sign-walk on k' to locate the critical points of k.  By
          Rolle's theorem, between any two roots of k there is at
          least one root of k', so #(real roots of k) <= #(crit pts) + 1.
      (4) Tangent-root check: a TANGENT root r of k satisfies BOTH
          k(r) = 0 AND k'(r) = 0.  The second condition forces r to
          be a critical point.  We evaluate |k(c)| at every critical
          point c found in [-10, 10]; if all are well above zero
          (>> 1e-8), no tangent root can hide there.  Combined with
          the FTA bound on TOTAL real roots, this rules out hidden
          tangent roots inside [-10, 10] entirely -- there is no
          "extra" root capacity for one to hide.
      (5) Refine each sign-change bracket via bisection.

    Returns
    -------
    roots : list of float
        Refined real roots of k on [-10, 10], in left-to-right order.
    """

    # ---- Variable declarations ----
    brackets_k = []                 # sign-change brackets of k (list of tuples)
    crit_brackets = []              # sign-change brackets of k' (list of tuples)
    crit_points = []                # refined critical points of k (list of float)
    roots = []                      # refined roots of k (list of float)
    has_tangent_root = False        # tangent-root check flag (bool)
    TANGENT_TOL = 1.0e-8            # |k(c)| threshold for tangent roots (float)

    print("=" * 76)
    print("FACTOR k(x) = x^5 - 9x^4 - x^3 + 17x^2 - 8x - 8")
    print("=" * 76)

    # ---- Step 1: theoretical upper bound (FTA) ----
    print("Step 1: By the Fundamental Theorem of Algebra, a degree-5")
    print("        polynomial has at most 5 real roots ON ALL OF R.")
    print()

    # ---- Step 2: locate sign-change brackets of k on [-10, 10] ----
    print("Step 2: Sign-walk for k(x) on [-10, 10] with grid spacing h=1e-3.")
    brackets_k = sign_change_brackets(k, -10.0, 10.0, h=1.0e-3)
    print(f"        Found {len(brackets_k)} sign-change brackets:")
    for (a, b) in brackets_k:
        print(f"          [{a:+.4f}, {b:+.4f}]   "
              f"k(a)={k(a):+.4e}, k(b)={k(b):+.4e}")
    print()

    # ---- Step 3: locate critical points of k via sign-walk on k' ----
    print("Step 3: Sign-walk for k'(x) on [-10, 10] to locate critical points.")
    crit_brackets = sign_change_brackets(k_prime, -10.0, 10.0, h=1.0e-3)
    print(f"        Found {len(crit_brackets)} critical-point brackets:")
    for (a, b) in crit_brackets:
        c, _ = bisection_method(k_prime, a, b, tol=1.0e-12, max_iter=200)
        crit_points.append(c)
        print(f"          critical point c ~ {c:+.10f}, "
              f"k(c) = {k(c):+.6e}")
    print()
    print("        By Rolle's theorem, between any two real roots of k")
    print("        there must lie a real root of k'.  Hence")
    print(f"        #(roots of k on [-10,10]) <= #(crit pts in [-10,10]) + 1 = "
          f"{len(crit_points) + 1}.")
    print()

    # ---- Step 4: tangent-root check ----
    # Logic: a tangent root r of k satisfies BOTH k(r) = 0 AND k'(r) = 0.
    # The second condition forces r to be a critical point of k.  So if
    # we evaluate k at every critical point in [-10, 10] and find that
    # |k(c)| is well above zero everywhere, no tangent root of k can lie
    # in [-10, 10].  Combined with the FTA bound (k has at most 5 real
    # roots TOTAL), this leaves no room for hidden tangent roots.
    print("Step 4: Tangent-root check.  A tangent root r of k requires both")
    print("        k(r) = 0 AND k'(r) = 0, so r MUST be a critical point.")
    print(f"        We require |k(c)| > {TANGENT_TOL:.0e} at every crit pt c.")
    for c in crit_points:
        if abs(k(c)) < TANGENT_TOL:
            print(f"        WARNING: |k({c:+.10f})| = {abs(k(c)):.3e} "
                  f"< {TANGENT_TOL:.0e} -- possible tangent root.")
            has_tangent_root = True
    if not has_tangent_root:
        min_kc = min(abs(k(c)) for c in crit_points)
        print(f"        Smallest |k(c)| over critical pts is {min_kc:.3e},")
        print("        many orders of magnitude above zero.  No tangent")
        print("        root of k lies in [-10, 10].  Combined with the FTA")
        print("        bound, every real root of k in [-10, 10] is simple")
        print("        (multiplicity 1) and produces a sign change.")
    print()

    # ---- Step 5: refine each sign-change bracket to a root ----
    print("Step 5: Refining each sign-change bracket via bisection.")
    for (a, b) in brackets_k:
        r, n = refine_root_bisection(k, (a, b), tol=1.0e-12)
        roots.append(r)
        print(f"          bracket [{a:+.4f}, {b:+.4f}] -> "
              f"root {r:+.12f}  (n_iter={n}, |k(r)|={abs(k(r)):.3e})")
    print()

    print(f"        CONCLUSION: k has exactly {len(roots)} real roots on [-10, 10].")
    print()

    return roots


def analyze_g():
    """
    Count the real roots of g(x) = e^(2x-1) - 2x^2 - 1/2 on [-10, 10].

    Strategy:
      (1) g is transcendental, so the FTA does not apply.  Instead,
          climb the derivative ladder until a derivative has a root
          structure we can solve in closed form:
                g(x)   = e^(2x-1) - 2x^2 - 1/2
                g'(x)  = 2 e^(2x-1) - 4x
                g''(x) = 4 e^(2x-1) - 4
          g''(x) = 0  iff  e^(2x-1) = 1  iff  x = 1/2  (one real root,
          analytically).  So by Rolle, g' has at most 2 real roots,
          and g has at most 3 real roots ON ALL OF R.
      (2) RIGOROUS monotonicity argument (NOT relying on a finite
          grid):  g'(x) attains its global minimum where g''(x) = 0,
          i.e. at x = 1/2.  At that point g'(1/2) = 2 - 2 = 0.  Thus
          g'(x) >= 0 everywhere on R, with equality only at x = 1/2.
          A monotone non-decreasing function can vanish on at most an
          interval, but since g is strictly increasing away from 1/2
          (g' > 0 there), g vanishes at AT MOST ONE point.
      (3) Direct evaluation: g(1/2) = e^0 - 1/2 - 1/2 = 0, and
          g'(1/2) = g''(1/2) = 0 with g'''(1/2) = 8 e^0 = 8 != 0.
          So x = 1/2 is a real root of g of multiplicity EXACTLY 3.
          Note: multiplicity 3 is ODD, so g does change sign at 1/2,
          and a sign-walk WILL detect it (unlike an even-multiplicity
          tangent root).
      (4) Numerical verification: sign-walk finds exactly one bracket,
          which bisects to x = 0.5 to within machine precision.
      (5) Optional Fixed Point Iteration diagnostic:
          Rewriting g(x)=0 as x = phi(x) gives
          phi(x) = (1 + ln(2x^2 + 1/2)) / 2.
          Although phi(1/2)=1/2, we have |phi'(1/2)| = 1, so the
          usual contraction theorem does not apply. We therefore do not
          use FPI as part of the proof. It is printed only as a diagnostic;
          the actual proof is the derivative and multiplicity argument above.
          Returns
    -------
    roots : list of float
        Refined real roots of g on [-10, 10] (a single root, x = 1/2).
    """

    # ---- Variable declarations ----
    brackets_g = []                 # sign-change brackets of g (list of tuples)
    roots = []                      # refined roots of g (list of float)
    fpi_root = 0.0                  # FPI cross-check result (double float)
    fpi_iters = 0                   # FPI iteration count (integer)

    print("=" * 76)
    print("FACTOR g(x) = e^(2x-1) - 2x^2 - 1/2")
    print("=" * 76)

    # ---- Step 1: theoretical upper bound via the Rolle ladder ----
    print("Step 1: Theoretical upper bound via the Rolle ladder.")
    print("        g''(x) = 4 e^(2x-1) - 4 = 0  iff  x = 1/2  (one root,")
    print("        analytically -- exp is monotone, so e^(2x-1) = 1 has")
    print("        a unique real solution).")
    print("        => by Rolle, g' has at most 2 real roots on R.")
    print("        => by Rolle, g  has at most 3 real roots on R.")
    print()

    # ---- Step 2: rigorous monotonicity of g (analytic, not sample-based) ----
    print("Step 2: Rigorous monotonicity argument (no finite-grid evidence).")
    print("        g''(x) = 4 e^(2x-1) - 4 < 0 for x < 1/2 and > 0 for")
    print("        x > 1/2, so g'(x) attains its global minimum at x = 1/2.")
    print(f"        At that point, g'(1/2) = 2*e^0 - 2 = {g_prime(0.5):+.4e}.")
    print("        Therefore g'(x) >= 0 for all x in R, with equality only")
    print("        at x = 1/2.  Hence g is monotone non-decreasing, and")
    print("        can vanish at AT MOST ONE point.")
    print()

    # ---- Step 3: exact root and multiplicity at x = 1/2 ----
    print("Step 3: Direct evaluation at x = 1/2:")
    print(f"          g(1/2)   = {g(0.5):+.6e}")
    print(f"          g'(1/2)  = {g_prime(0.5):+.6e}")
    print(f"          g''(1/2) = {g_double_prime(0.5):+.6e}")
    print(f"          g'''(1/2) = 8 e^0 = {8.0:+.6e}  (nonzero)")
    print("        So x = 1/2 is a root of g of multiplicity EXACTLY 3.")
    print("        Multiplicity 3 is ODD, so g changes sign at 1/2 and")
    print("        the root WILL be detected by a sign-walk.")
    print()

    # ---- Step 4: numerical verification via sign-walk on g ----
    print("Step 4: Numerical verification via sign-walk on g, h=1e-3.")
    brackets_g = sign_change_brackets(g, -10.0, 10.0, h=1.0e-3)
    print(f"        Found {len(brackets_g)} sign-change bracket(s):")
    for (a, b) in brackets_g:
        print(f"          [{a:+.4f}, {b:+.4f}]   "
              f"g(a)={g(a):+.4e}, g(b)={g(b):+.4e}")
    print()

    print("Step 5: Refining the bracket via bisection.")
    for (a, b) in brackets_g:
        r, n = refine_root_bisection(g, (a, b), tol=1.0e-12)
        roots.append(r)
        print(f"          bracket [{a:+.4f}, {b:+.4f}] -> "
              f"root {r:+.12f}  (n_iter={n}, |g(r)|={abs(g(r)):.3e})")
    print()

    # ---- Step 6: independent cross-check via Fixed Point Iteration ----
    # FPI provides an independent confirmation that x = 1/2 is genuinely
    # a root.  We rewrite g(x) = 0 as x = phi(x):
    #     e^(2x-1) - 2x^2 - 1/2 = 0
    # =>  e^(2x-1) = 2x^2 + 1/2
    # =>  2x - 1 = ln(2x^2 + 1/2)
    # =>  x = (1 + ln(2x^2 + 1/2)) / 2  =:  phi(x)
    # Note phi is well defined for all x (since 2x^2 + 1/2 > 0).
    # phi(1/2) = (1 + ln(1))/2 = 1/2, so x = 1/2 is a fixed point.
    print("Step 6: Independent cross-check via Fixed Point Iteration.")
    print("        Rewrite g(x) = 0 as x = phi(x) where")
    print("            phi(x) = (1 + ln(2x^2 + 1/2)) / 2.")
    print("        phi(1/2) = (1 + ln(1))/2 = 1/2, so 1/2 is a fixed point.")

    def phi_g(x):
        # FPI map for g: x = (1 + ln(2x^2 + 1/2))/2.
        return (1.0 + math.log(2.0*x**2 + 0.5)) / 2.0

    fpi_root, fpi_iters, fpi_converged = fixed_point_iteration(
        phi_g, 0.45, tol=1.0e-8, max_iter=10000, return_converged=True
    )
    print(f"        FPI from x_0 = 0.45 (tol=1e-8, max_iter=10000):")
    print(f"            final iterate x = {fpi_root:.10f} after {fpi_iters} iters")
    print(f"            stopped by tolerance? {fpi_converged}")
    print(f"            |g(fpi_root)| = {abs(g(fpi_root)):.3e}")
    print("        Important: this is NOT a contraction proof.  Here |phi'(1/2)|")
    print("        = 1, so the standard fixed-point convergence theorem does not")
    print("        apply.  FPI is included only as a weak optional diagnostic; the")
    print("        actual proof for g is the derivative/multiplicity argument above.")
    print()

    print(f"        CONCLUSION: g has exactly {len(roots)} real root on [-10, 10],")
    print("        located at x = 1/2 with multiplicity 3.")
    print()

    return roots


def make_numeric_root(factor, value, residual_func, description=""):
    """
    Package a numerically resolved root for reporting and overlap checking.

    The `value` is a genuine floating-point approximation to the root, so it
    is safe to evaluate the full product f at that value for a residual.
    """
    return {
        "factor": factor,
        "value": float(value),
        "label": f"{float(value):+.20e}",
        "description": description,
        "symbolic": False,
        "residual_func": residual_func,
    }


def make_symbolic_h_root(center, side, delta):
    """
    Package one of the sub-floating-point h-roots near +/- 3*pi.

    These roots are mathematically distinct, but the offset is much smaller
    than one double-precision spacing unit near +/- 3*pi.  Therefore the two
    roots flanking the same center cannot be represented as distinct Python
    floats.  We store a symbolic label and use the center only as a sorting /
    coarse overlap-check location.
    """
    sign_string = "-" if side < 0 else "+"
    center_label = "-3*pi" if center < 0 else "+3*pi"
    return {
        "factor": "h",
        "value": float(center),
        "label": f"{center_label} {sign_string} delta, delta ~= {delta:.3e}",
        "description": "analytic h-root below double-precision resolution",
        "symbolic": True,
        "center": float(center),
        "side": int(side),
        "delta": float(delta),
        "residual_func": None,
    }


def h_prime_sign_change_brackets(a, b, grid_spacing=1.0e-4):
    """
    Locate sign-change brackets for h' on a subinterval.

    This helper is only used as computational evidence that h has no extra
    roots away from the expected neighborhoods.  It does not replace the
    analytic argument; it gives a reproducible monotonicity sanity check.
    """
    return sign_change_brackets(h_prime, a, b, h=grid_spacing)


def analyze_h():
    """
    Count the real roots of h(x) = cos(x) - e^(-x^2) + 1 on [-10, 10].

    The mathematical answer is 8.  Four are ordinary floating-point-resolvable
    roots near +/- pi.  Four more are real roots near +/- 3*pi whose offsets
    from +/- 3*pi are about 7.3e-20, far below double-precision spacing near
    9.42.  Those four roots are stored symbolically rather than as fake
    distinct floats.
    """

    # ---- Variable declarations ----
    candidates = []                 # odd multiples of pi in [-10, 10]
    brackets_h = []                 # sign-change brackets of h
    roots = []                      # root records, not just floats
    n_resolvable = 0                # # of FP-resolvable roots
    n_unresolvable = 0              # # of analytic-only roots

    print("=" * 76)
    print("FACTOR h(x) = cos(x) - e^(-x^2) + 1")
    print("=" * 76)

    # ---- Step 1: structural decomposition ----
    print("Step 1: Rewrite h(x) = (1 + cos(x)) - e^(-x^2).")
    print("        First term: >= 0, vanishes iff x is an odd multiple of pi.")
    print("        Second term: in (0, 1], strictly positive.")
    print("        Hence roots can occur only where the nonnegative cosine")
    print("        term is as small as the Gaussian term, which forces x")
    print("        to be near an odd multiple of pi.")
    print()

    # ---- Step 2: Taylor-derived root locations near each candidate ----
    print("Step 2: Taylor expansion near each odd multiple of pi.")
    print("        Let c = (2k+1)*pi and u = x - c.  Then")
    print("            1 + cos(c+u) = 1 - cos(u) = u^2/2 - u^4/24 + ...,")
    print("            e^(-(c+u)^2) = e^(-c^2) * e^(-2cu-u^2).")
    print("        Therefore h(c+u) = u^2/2 - e^(-c^2) + higher-order terms.")
    print("        To leading order, the two roots flanking c satisfy")
    print("            u = +/- sqrt(2 * e^(-c^2)).")
    print()

    # ---- Step 3: enumerate candidates and evaluate FP resolvability ----
    print("Step 3: Odd multiples of pi inside [-10, 10] and their offsets:")
    for n in [-3, -1, 1, 3]:
        c = n * math.pi
        c_sq = c * c
        u_offset = math.sqrt(2.0 * math.exp(-c_sq))
        fp_spacing = abs(np.spacing(c))
        resolvable = u_offset > 10.0 * fp_spacing
        candidates.append((n, c, u_offset, resolvable))
        status = "RESOLVABLE" if resolvable else "BELOW FP RESOLUTION"
        print(f"          c = {c:+.6f}  (c^2 = {c_sq:.3f})")
        print(f"            offset u ~ {u_offset:.3e}, "
              f"FP spacing near c ~ {fp_spacing:.3e}")
        print(f"            ratio u / FP spacing = "
              f"{u_offset/fp_spacing:.3e}  -> {status}")
    print("        (+/- 5*pi ~= +/- 15.708 lies outside [-10, 10].)")
    print()

    # ---- Step 4: computational no-extra-roots sanity check ----
    print("Step 4: No-extra-roots sanity check for h.")
    print("        Since h is even, it is enough to inspect [0, 10].")
    print("        We partition [0, 10] at the odd multiples of pi where the")
    print("        cosine bump bottoms out:")
    print("            (0, pi), (pi, 2*pi), (2*pi, 3*pi), (3*pi, 10).")
    print("        On each of these subintervals we sample h at its endpoints")
    print("        and at every interior critical point of h'.  The min/max of")
    print("        these h-values is reported below.  The known dips of h ARE")
    print("        included near +/- pi (handled in Step 5) and near +/- 3*pi")
    print("        (handled symbolically in Step 7); the purpose of this step")
    print("        is to give reproducible computational evidence that h does")
    print("        not have any UNEXPECTED roots away from those known dips.")

    positive_intervals = [
        (0.0, math.pi, "(0, pi)"),
        (math.pi, 2.0 * math.pi, "(pi, 2*pi)"),
        (2.0 * math.pi, 3.0 * math.pi, "(2*pi, 3*pi)"),
        (3.0 * math.pi, 10.0, "(3*pi, 10)"),
    ]
    for a, b, label in positive_intervals:
        # Avoid endpoints where h is extremely close to zero at odd multiples.
        eps = 1.0e-6
        aa = a + eps
        bb = b - eps
        crit_brackets = h_prime_sign_change_brackets(aa, bb, grid_spacing=1.0e-4)
        crit_points = []
        for bracket in crit_brackets:
            ccrit, _ = bisection_method(h_prime, bracket[0], bracket[1],
                                        tol=1.0e-12, max_iter=200)
            crit_points.append(ccrit)
        values_to_check = [aa] + crit_points + [bb]
        min_h = min(h(x) for x in values_to_check)
        max_h = max(h(x) for x in values_to_check)
        print(f"          {label}: found {len(crit_points)} h' critical point(s); "
              f"sampled h-range [{min_h:+.3e}, {max_h:+.3e}]")
    print("        Reading the ranges: small negative min h-values inside")
    print("        (0, pi) and (pi, 2*pi) come from the known dips near +/- pi")
    print("        that Step 5 already accounts for; the strictly-positive")
    print("        ranges on (2*pi, 3*pi) and (3*pi, 10) support that h has")
    print("        no surprise sign changes elsewhere on [0, 10] (and hence,")
    print("        by even symmetry, on [-10, 0]).")
    print()

    # ---- Step 5: sign-walk on h (will catch the 4 resolvable roots) ----
    print("Step 5: Sign-walk for h on [-10, 10] with grid spacing h=1e-3.")
    brackets_h = sign_change_brackets(h, -10.0, 10.0, h=1.0e-3)
    print(f"        Found {len(brackets_h)} sign-change brackets "
          f"(all near +/- pi as expected):")
    for (a, b) in brackets_h:
        print(f"          [{a:+.4f}, {b:+.4f}]   "
              f"h(a)={h(a):+.4e}, h(b)={h(b):+.4e}")
    print()

    # ---- Step 6: refine the resolvable brackets via bisection ----
    print("Step 6: Refining each sign-change bracket via bisection.")
    for (a, b) in brackets_h:
        r, n = refine_root_bisection(h, (a, b), tol=1.0e-12)
        roots.append(make_numeric_root("h", r, h, "floating-point-resolvable h-root"))
        n_resolvable += 1
        print(f"          bracket [{a:+.4f}, {b:+.4f}] -> "
              f"root {r:+.12f}  (n_iter={n}, |h(r)|={abs(h(r)):.3e})")
    print()

    # ---- Step 7: handle the unresolvable roots near +/- 3*pi ----
    print("Step 7: The 4 mathematical roots near +/- 3*pi cannot be")
    print("        isolated in floating point (offset ~7e-20 << FP spacing")
    print("        ~1.8e-15).  The Gaussian term e^(-9*pi^2) is representable,")
    print("        but it is lost to rounding/cancellation when combined with")
    print("        the 1 + cos(x) expression at double precision.")
    for n in [-3, 3]:
        c = n * math.pi
        true_center_value = -math.exp(-(n * math.pi) ** 2)
        print(f"          h({c:+.10f}) computes to {h(c):+.3e} "
              f"(mathematically near the center: {true_center_value:.3e})")
    print()
    print("        We record these roots symbolically instead of pretending")
    print("        c-u and c+u are distinct Python floats:")
    for n in [-3, 3]:
        c = n * math.pi
        u_offset = math.sqrt(2.0 * math.exp(-c * c))
        roots.append(make_symbolic_h_root(c, -1, u_offset))
        roots.append(make_symbolic_h_root(c, +1, u_offset))
        n_unresolvable += 2
        center_label = "-3*pi" if n < 0 else "+3*pi"
        print(f"          analytic root pair near c = {center_label}:")
        print(f"            left:  {center_label} - delta")
        print(f"            right: {center_label} + delta")
        print(f"            delta ~= {u_offset:.3e}")
    print()

    roots.sort(key=lambda record: (record["value"], record.get("side", 0)))

    print(f"        CONCLUSION: h has exactly {len(roots)} real roots on")
    print(f"        [-10, 10] mathematically: {n_resolvable} resolvable")
    print(f"        in floating point + {n_unresolvable} below FP resolution.")
    print()

    return roots


# =============================================================================
# SECTION D: OVERLAP / DOUBLE-COUNT CHECK
# -----------------------------------------------------------------------------
# A point x is a root of f = g*h*k iff x is a root of at least one
# factor.  If any x-value is shared by two or more factors, it must be
# counted only once in the total.  We check this by comparing all
# (factor, root) pairs at floating-point precision.
# =============================================================================


def root_record_from_value(value, factor):
    """
    Convert a plain float root into the same dictionary format used for
    symbolic h-roots.  This keeps the overlap-checking code uniform.
    """
    residual_functions = {"g": g, "h": h, "k": k}
    return make_numeric_root(
        factor, value, residual_functions.get(factor), f"numeric root of {factor}"
    )


def check_overlaps(roots_g, roots_h, roots_k, tol=1.0e-6):
    """
    Inspect the union of roots from g, h, k for cross-factor overlaps.

    Same-factor symbolic duplicates are intentionally preserved.  This matters
    for h near +/- 3*pi: the two roots flanking the same center are distinct
    real numbers even though their nearest double-precision representations
    are identical.
    """

    # ---- Convert all roots to root-record dictionaries ----
    tagged_g = [root_record_from_value(r, "g") for r in roots_g]
    tagged_h = [r if isinstance(r, dict) else root_record_from_value(r, "h")
                for r in roots_h]
    tagged_k = [root_record_from_value(r, "k") for r in roots_k]

    all_tagged = tagged_g + tagged_h + tagged_k
    all_tagged.sort(key=lambda record: (record["value"], record["factor"],
                                        record.get("side", 0)))

    overlaps = []
    distinct_roots = []
    used = [False] * len(all_tagged)

    # Walk through the sorted list; for each unmerged entry, scan for any
    # later entries from a DIFFERENT factor that fall within tol.
    for i in range(len(all_tagged)):
        if used[i]:
            continue

        root_i = all_tagged[i]
        x_i = root_i["value"]
        factor_i = root_i["factor"]
        merged_roots = [root_i]
        merged_factors = [factor_i]
        used[i] = True

        for j in range(i + 1, len(all_tagged)):
            if used[j]:
                continue
            root_j = all_tagged[j]
            x_j = root_j["value"]
            factor_j = root_j["factor"]

            # Stop scanning once the gap exceeds tol (list is sorted).
            if x_j - x_i > tol:
                break

            # Same-factor near-duplicates are not cross-factor overlaps.
            if factor_j == factor_i or factor_j in merged_factors:
                continue

            merged_roots.append(root_j)
            merged_factors.append(factor_j)
            used[j] = True

        # If there is a genuine cross-factor overlap, represent the merged
        # root by the first record but note the overlap for reporting.
        representative = merged_roots[0].copy()
        if len(set(merged_factors)) > 1:
            x_avg = sum(root["value"] for root in merged_roots) / len(merged_roots)
            representative["value"] = x_avg
            representative["label"] = f"{x_avg:+.20e}"
            representative["description"] = "cross-factor merged root"
            overlaps.append((x_avg, merged_factors))

        distinct_roots.append(representative)

    distinct_roots.sort(key=lambda record: (record["value"], record["factor"],
                                            record.get("side", 0)))
    return distinct_roots, overlaps


# =============================================================================
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
# Running this file from the command line executes the full Part 3
# investigation: theoretical analysis + numerical evidence for each
# factor, overlap check, and the final count.
# =============================================================================


if __name__ == "__main__":

    print()
    print("#" * 76)
    print("# PA2 PART 3 -- HOW MANY REAL ROOTS DOES f HAVE ON [-10, 10]?")
    print("#" * 76)
    print()
    print("f(x) = ( e^(2x-1) - 2x^2 - 1/2 )")
    print("     * ( cos(x) - e^(-x^2) + 1 )")
    print("     * ( x^5 - 9x^4 - x^3 + 17x^2 - 8x - 8 )")
    print()
    print("By the Zero Product Property, the real roots of f are the union")
    print("of the real roots of its three factors g, h, k.  We analyze each")
    print("factor in turn (Part 1 methods + analytic theory) and then")
    print("check for shared x-values.  Fixed Point Iteration is discussed")
    print("only as an optional diagnostic where it is not theorem-valid.")
    print()

    # ---- Per-factor analyses ----
    roots_k = analyze_k()
    roots_g = analyze_g()
    roots_h = analyze_h()

    # ---- Overlap / double-count check ----
    print("=" * 76)
    print("OVERLAP CHECK (no x-value should be a root of more than one factor)")
    print("=" * 76)
    distinct_roots, overlaps = check_overlaps(roots_g, roots_h, roots_k,
                                              tol=1.0e-6)
    if not overlaps:
        print("No overlaps: every refined root belongs to exactly one factor.")
    else:
        print("OVERLAPS DETECTED:")
        for (x, fs) in overlaps:
            print(f"  x = {x:+.10f} is a root of factors: {', '.join(fs)}")
    print()

    # ---- Final tally ----
    print("=" * 76)
    print("FINAL TALLY")
    print("=" * 76)
    # Count h roots by category, deriving the breakdown from the actual
    # records rather than hardcoding "4 + 4" -- that way the tally line
    # remains correct if anyone later changes the analysis interval.
    n_h_symbolic = sum(
        1 for r in roots_h if isinstance(r, dict) and r.get("symbolic", False)
    )
    n_h_numeric = len(roots_h) - n_h_symbolic
    print(f"  Roots of k on [-10, 10]: {len(roots_k)}")
    print(f"  Roots of g on [-10, 10]: {len(roots_g)} "
          f"(multiplicity 3 at x = 1/2)")
    print(f"  Roots of h on [-10, 10]: {len(roots_h)} "
          f"({n_h_numeric} resolvable + {n_h_symbolic} sub-FP near +/- 3*pi)")
    print(f"  Overlaps between factors: {len(overlaps)}")
    print(f"  ---------------------------------------------")
    n_total = len(roots_k) + len(roots_g) + len(roots_h) - len(overlaps)
    print(f"  TOTAL DISTINCT REAL ROOTS OF f ON [-10, 10]: {n_total}")
    print()
    print("Distinct roots, sorted:")
    for root in distinct_roots:
        if root.get("symbolic", False):
            print(f"  x = {root['label']}   ({root['description']})")
        else:
            x_value = root["value"]
            residual_function = root.get("residual_func")
            if residual_function is None:
                residual_text = f"|f(x)| = {abs(f(x_value)):.3e}"
            else:
                residual_text = (
                    f"|{root['factor']}(x)| = {abs(residual_function(x_value)):.3e}"
                )
            print(f"  x = {x_value:+.20e}   ({residual_text})")
    print()

    print("#" * 76)
    print("# END OF PART 3 INVESTIGATION")
    print("#" * 76)