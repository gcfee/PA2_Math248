###############################################################################
# Program: PA2.Part.2.py
# Author(s): Griffin Fee, Jack Zettlemoyer, Kai Behrens, Jacob Fitzmaurice
# Date: 5/3/26
# Purpose: This program implements the testing harness for Part 2 of
#          Math 248, Programming Assignment 2. It contains the three
#          root-finding methods from Part 1 (Newton's, Bisection,
#          Secant) directly in this file, runs them on a set of
#          carefully chosen test cases, computes absolute errors
#          against known true roots, estimates the experimental order
#          of convergence (EOC), and prints professionally formatted
#          tables for use in the write-up.
#
#          To use, run this file directly from the command line:
#               >> python3 PA2.Part.2.py
#          All seven required test cases (i)-(vii) from the assignment
#          will be executed in order, and their results printed to the
#          screen.
###############################################################################
# Inputs:  None (test cases are hard-coded for reproducibility).
# Outputs: Formatted convergence tables printed to stdout for each test
#          case, including iteration index, iterate value, absolute
#          error, and experimental order of convergence per iteration.
###############################################################################

import math                                  # for log, sqrt, pi, etc.


# =============================================================================
# PART 1 METHODS: NEWTON, BISECTION, SECANT
# -----------------------------------------------------------------------------
# These three functions are reproduced verbatim from PA2.Part.1.py so
# that this file is self-contained -- a grader only needs PA2.Part.2.py
# to reproduce every Part 2 result.
# =============================================================================


def newtons_method(f, f_prime, x0, tol=1e-10, max_iter=100, return_iterates=False):
    """
    Newton's Method for root-finding.

    Computes an approximate root of f(x) = 0 using the iteration:
        x_{n+1} = x_n - f(x_n) / f'(x_n)

    Parameters
    ----------
    f : callable
        The function whose root we seek. Must accept a single float
        and return a single float.
    f_prime : callable
        The derivative of f. Must accept a single float and return
        a single float.
    x0 : float
        The initial guess for the root.
    tol : float, optional
        Convergence tolerance for |x_{n+1} - x_n|. Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 100.
    return_iterates : bool, optional
        If True, the function returns the full sequence of iterates
        and the residuals |f(x_n)| at each step. Default is False.

    Returns
    -------
    root : float
        The approximate root found by the method.
    num_iterations : int
        The number of iterations performed.
    iterates : list of float (only if return_iterates is True)
        The sequence [x0, x1, x2, ...] of all iterates.
    residuals : list of float (only if return_iterates is True)
        The sequence [|f(x0)|, |f(x1)|, ...] of residual magnitudes.

    Raises
    ------
    ZeroDivisionError
        If f'(x_n) = 0 at any iterate, division by zero is not possible
        and the method raises an error.
    """

    # ---- Variable declarations ----
    x_current = float(x0)          # current iterate x_n (double float)
    x_next = 0.0                   # next iterate x_{n+1} (double float)
    f_val = 0.0                    # f(x_n) at current iterate (double float)
    f_prime_val = 0.0              # f'(x_n) at current iterate (double float)
    diff = 0.0                     # |x_{n+1} - x_n| (double float)
    iteration_count = 0            # number of iterations performed (integer)

    # If the user wants the full history, initialize storage lists
    if return_iterates:
        iterates_list = [x_current]              # stores every x_n
        residuals_list = [abs(f(x_current))]     # stores |f(x_n)| at each step

    # ---- Main iteration loop ----
    for iteration_count in range(1, max_iter + 1):

        # Step 1: Evaluate f and f' at the current iterate
        f_val = f(x_current)
        f_prime_val = f_prime(x_current)

        # Step 2: Check for division by zero
        #   If f'(x_n) = 0, Newton's method cannot proceed because the
        #   tangent line is horizontal and has no x-intercept.
        if f_prime_val == 0.0:
            raise ZeroDivisionError(
                f"Newton's method failed: f'(x) = 0 at x = {x_current} "
                f"(iteration {iteration_count}). The tangent line is "
                f"horizontal, so no next iterate can be computed."
            )

        # Step 3: Compute the next iterate using Newton's formula
        #   x_{n+1} = x_n - f(x_n) / f'(x_n)
        x_next = x_current - f_val / f_prime_val

        # Step 4: Compute the step size |x_{n+1} - x_n|
        diff = abs(x_current - x_next)

        # Step 5: Record the iterate if the user requested the history
        if return_iterates:
            iterates_list.append(x_next)
            residuals_list.append(abs(f(x_next)))

        # Step 6: Update x_current for the next iteration
        x_current = x_next

        # Step 7: Check for convergence
        #   If the step size is below the tolerance, we consider the
        #   method to have converged and stop early.
        if diff < tol:
            break

    # ---- Return results ----
    if return_iterates:
        return x_current, iteration_count, iterates_list, residuals_list
    else:
        return x_current, iteration_count


# =============================================================================


def bisection_method(f, a, b, tol=1e-10, max_iter=100, return_iterates=False):
    """
    Bisection Method for root-finding.

    Computes an approximate root of f(x) = 0 by repeatedly halving
    an interval [a, b] that contains a sign change, keeping the
    sub-interval where the sign change persists.

    Parameters
    ----------
    f : callable
        The function whose root we seek. Must accept a single float
        and return a single float.
    a : float
        The left endpoint of the starting interval.
    b : float
        The right endpoint of the starting interval.
    tol : float, optional
        Convergence tolerance for the interval width (b - a).
        Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 100.
    return_iterates : bool, optional
        If True, the function returns the full sequence of midpoint
        iterates and the residuals |f(m)| at each step. Default is False.

    Returns
    -------
    root : float
        The approximate root (the final midpoint).
    num_iterations : int
        The number of iterations performed.
    iterates : list of float (only if return_iterates is True)
        The sequence [m1, m2, m3, ...] of all midpoint iterates.
    residuals : list of float (only if return_iterates is True)
        The sequence [|f(m1)|, |f(m2)|, ...] of residual magnitudes.

    Raises
    ------
    ValueError
        If the sign-change hypothesis f(a)*f(b) < 0 fails, bisection
        cannot guarantee a root exists and the method raises an error.
    """

    # ---- Variable declarations ----
    a = float(a)                   # left endpoint of current interval (double float)
    b = float(b)                   # right endpoint of current interval (double float)
    m = 0.0                        # midpoint of current interval (double float)
    f_a = f(a)                     # f evaluated at left endpoint (double float)
    f_b = f(b)                     # f evaluated at right endpoint (double float)
    f_m = 0.0                      # f evaluated at midpoint (double float)
    iteration_count = 0            # number of iterations performed (integer)

    # ---- Check the sign-change hypothesis ----
    #   The Intermediate Value Theorem guarantees a root exists in [a, b]
    #   only if f(a) and f(b) have opposite signs, i.e., f(a)*f(b) < 0.
    if f_a * f_b > 0:
        raise ValueError(
            f"Bisection method requires f(a) * f(b) < 0, but "
            f"f({a})*f({b}) = {f_a * f_b:.6e} >= 0. "
            f"No sign change detected on [{a}, {b}]."
        )

    # ---- Handle edge case: endpoint is already a root ----
    if f_a == 0:
        return (a, 0, [a], [0.0]) if return_iterates else (a, 0)

    if f_b == 0:
        return (b, 0, [b], [0.0]) if return_iterates else (b, 0)

    # If the user wants the full history, initialize storage lists
    if return_iterates:
        iterates_list = []                       # stores every midpoint m
        residuals_list = []                      # stores |f(m)| at each step

    # ---- Main iteration loop ----
    for iteration_count in range(1, max_iter + 1):

        # Step 1: Compute the midpoint of the current interval
        m = (a + b) / 2.0

        # Step 2: Evaluate f at the midpoint
        f_m = f(m)

        # Step 3: Record the iterate if the user requested the history
        if return_iterates:
            iterates_list.append(m)
            residuals_list.append(abs(f_m))

        # Step 4: Check if we landed exactly on a root
        #   If f(m) = 0, we have found the root exactly and can stop.
        if f_m == 0:
            break

        # Step 5: Decide which sub-interval contains the root
        #   If f(a) and f(m) have opposite signs, the root lies in [a, m],
        #   so we move b inward. Otherwise, the root lies in [m, b],
        #   so we move a inward.
        if f_a * f_m < 0:
            b = m
            f_b = f_m
        else:
            a = m
            f_a = f_m

        # Step 6: Check for convergence
        #   Once the interval width is below the tolerance, the root
        #   is located precisely enough and we stop.
        if (b - a) < tol:
            break

    # ---- Return results ----
    if return_iterates:
        return m, iteration_count, iterates_list, residuals_list
    else:
        return m, iteration_count


# =============================================================================


def secant_method(f, x0, x1, tol=1e-10, max_iter=100, return_iterates=False):
    """
    Find a root of f(x) = 0 using the secant method.

    The secant method approximates the derivative in Newton's method with a
    finite difference, using the two most recent iterates to construct a
    secant line whose x-intercept becomes the next iterate:

        x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    Unlike Newton's method, no derivative evaluation is required, making
    this suitable for functions where f'(x) is expensive or unavailable.
    The method converges with order approximately 1.618 (the golden
    ratio) near a simple root.

    Parameters
    ----------
    f : callable
        The function whose root is sought. Must accept and return a float.
    x0 : float
        First initial guess.
    x1 : float
        Second initial guess (should differ from x0).
    tol : float, optional
        Convergence tolerance on the step size |x_{n+1} - x_n|.
        Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations before stopping. Default is 100.
    return_iterates : bool, optional
        If True, return the full history of iterates and residuals.
        Default is False.

    Returns
    -------
    x_current : float
        The approximate root.
    iteration_count : int
        Number of iterations performed.
    iterates_list : list of float
        (Only if return_iterates=True) All iterates from x0 through the
        final x_n.
    residuals_list : list of float
        (Only if return_iterates=True) |f(x_n)| at each iterate.

    Raises
    ------
    ZeroDivisionError
        If f(x_n) = f(x_{n-1}) at any step, making the secant line
        horizontal and the next iterate undefined.
    """

    # ---- Variable declarations ----
    x_current = float(x1)          # current iterate x_n (double float)
    x_previous = float(x0)         # previous iterate x_{n-1} (double float)
    x_next = 0.0                   # next iterate x_{n+1} (double float)
    f_val_curr = 0.0               # f(x_n) at current iterate (double float)
    f_val_prev = 0.0               # f(x_{n-1}) at previous iterate (double float)
    diff = 0.0                     # |x_{n+1} - x_n| (double float)
    iteration_count = 0            # number of iterations performed (integer)

    # If the user wants the full history, initialize storage lists
    if return_iterates:
        iterates_list = [x_previous, x_current]
        residuals_list = [abs(f(x_previous)), abs(f(x_current))]

    # ---- Main iteration loop ----
    for iteration_count in range(1, max_iter + 1):

        # Step 1: Evaluate f at the current and previous iterate
        f_val_curr = f(x_current)
        f_val_prev = f(x_previous)

        # Step 2: Check for division by zero
        #   If f(x_n) = f(x_{n-1}), the secant method cannot proceed
        #   because the secant line is horizontal and has no x-intercept.
        if f_val_curr - f_val_prev == 0.0:
            raise ZeroDivisionError(
                f"Secant method failed: f(x) - f(x_-1) = 0 at x = {x_current} "
                f"(iteration {iteration_count}). The secant line is "
                f"horizontal, so no next iterate can be computed."
            )

        # Step 3: Compute the next iterate using the secant formula
        #   x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        x_next = x_current - f_val_curr * (
            (x_current - x_previous) / (f_val_curr - f_val_prev)
        )

        # Step 4: Compute the step size
        diff = abs(x_current - x_next)

        # Step 5: Record the iterate if the user requested the history
        if return_iterates:
            iterates_list.append(x_next)
            residuals_list.append(abs(f(x_next)))

        # Step 6: Update x_current and x_previous for the next iteration
        x_previous = x_current
        x_current = x_next

        # Step 7: Check for convergence
        #   If the step size is below the tolerance, we consider the
        #   method to have converged and stop early.
        if diff < tol:
            break

    # ---- Return results ----
    if return_iterates:
        return x_current, iteration_count, iterates_list, residuals_list
    else:
        return x_current, iteration_count


# =============================================================================
# LAYER 1: SHARED TESTING INFRASTRUCTURE
# -----------------------------------------------------------------------------
# The three helpers below are used by every test case. They take the
# raw output of a root-finding method (a list of iterates) and turn it
# into the quantities required by the PA2 write-up:
#   * absolute_errors        - |x_k - r| for each iterate
#   * experimental_order     - estimated order of convergence p_k for
#                              each interior iterate, using the PA2
#                              estimator:
#                                p ~ ln(e_{k+1}/e_k) / ln(e_k/e_{k-1})
#   * print_convergence_table - formatted table combining all of the above
# =============================================================================


def absolute_errors(iterates, true_root):
    """
    Compute the absolute error |x_k - r| for each iterate in a sequence.

    Parameters
    ----------
    iterates : list of float
        The sequence [x_0, x_1, x_2, ...] of iterates produced by a
        root-finding method.
    true_root : float
        The known true root r of the function. For PA2, all test
        functions are chosen so that r is known exactly (e.g.,
        r = sqrt(2) for f(x) = x^2 - 2).

    Returns
    -------
    errors : list of float
        The sequence [|x_0 - r|, |x_1 - r|, |x_2 - r|, ...] of
        absolute errors, in the same order as `iterates`.
    """

    # ---- Variable declarations ----
    errors = []                    # output list of absolute errors (list of float)
    r = float(true_root)           # local copy of the true root (double float)

    # Walk through every iterate and compute its distance to the true root
    for x_k in iterates:
        errors.append(abs(x_k - r))

    return errors


# =============================================================================


def experimental_order(errors):
    """
    Estimate the experimental order of convergence (EOC) from a list of
    absolute errors, using the estimator from the PA2 assignment:

        p_k ~  ln(e_{k+1} / e_k) / ln(e_k / e_{k-1})

    A separate estimate p_k is computed for each interior triple
    (e_{k-1}, e_k, e_{k+1}). Triples that contain a zero or negative
    error are skipped (recorded as None in the output) because the
    logarithm of zero is undefined and such errors are not "reliably
    computed" in the sense of the PA2 spec.

    Parameters
    ----------
    errors : list of float
        The sequence [e_0, e_1, e_2, ...] of absolute errors.
        Must contain at least 3 entries to produce any EOC estimate.

    Returns
    -------
    eocs : list of (float or None)
        A list of length len(errors), where:
          * eocs[0] = None        (no e_{-1} exists)
          * eocs[k] = p_k         for 1 <= k <= len(errors) - 2,
                                  computed from (e_{k-1}, e_k, e_{k+1}),
                                  or None if any of those three errors
                                  are not strictly positive.
          * eocs[-1] = None       (no e_{n+1} exists)
        This shape makes alignment with the iterate/error columns
        trivial when printing the convergence table.
    """

    # ---- Variable declarations ----
    n = len(errors)                # total number of error values (integer)
    eocs = [None] * n              # output list, same length as errors (list)
    e_km1 = 0.0                    # e_{k-1}, the previous error (double float)
    e_k = 0.0                      # e_k,     the current error  (double float)
    e_kp1 = 0.0                    # e_{k+1}, the next error     (double float)
    numerator = 0.0                # ln(e_{k+1}/e_k)             (double float)
    denominator = 0.0              # ln(e_k/e_{k-1})             (double float)

    # We need at least three errors to form even a single triple.
    # If we don't have that many, all entries remain None.
    if n < 3:
        return eocs

    # Loop over every interior index k that has both a predecessor
    # and a successor available (1 <= k <= n - 2).
    for k in range(1, n - 1):

        # Step 1: Pull out the relevant triple of errors
        e_km1 = errors[k - 1]
        e_k   = errors[k]
        e_kp1 = errors[k + 1]

        # Step 2: Skip triples with zero or negative errors.
        #   The PA2 spec instructs us to only use iterations whose
        #   errors are nonzero and reliably computed. The log of zero
        #   is -infinity, and a zero in the denominator ratio would
        #   blow up the formula entirely.
        if e_km1 <= 0.0 or e_k <= 0.0 or e_kp1 <= 0.0:
            continue

        # Step 3: Skip the (rare) degenerate case where two consecutive
        #   errors are exactly equal. This makes ln(e_k/e_{k-1}) = 0
        #   and the formula divides by zero. In practice this only
        #   happens for stalled iterations, which are not informative.
        if e_k == e_km1 or e_kp1 == e_k:
            continue

        # Step 4: Apply the PA2 EOC formula
        #   p ~ ln(e_{k+1}/e_k) / ln(e_k/e_{k-1})
        numerator   = math.log(e_kp1 / e_k)
        denominator = math.log(e_k / e_km1)

        # One last guard: even after the equality check above,
        # floating-point cancellation could still produce a
        # near-zero denominator. Skip if so.
        if denominator == 0.0:
            continue

        eocs[k] = numerator / denominator

    return eocs


# =============================================================================


def print_convergence_table(iterates, errors, eocs,
                            iterate_label="x_n",
                            include_ratio=False):
    """
    Print a formatted convergence table for a single test case.

    The table contains one row per iterate, with columns:
        k          -- iteration index
        x_n        -- the iterate value (column header customizable)
        |x_n - r|  -- absolute error
        p_k        -- experimental order of convergence
        e_{k+1}/e_k -- successive-error ratio (only if include_ratio=True;
                       required for the bisection EOC test, item (vii))

    Parameters
    ----------
    iterates : list of float
        Sequence of iterates [x_0, x_1, ...].
    errors : list of float
        Absolute errors aligned with `iterates`.
    eocs : list of (float or None)
        EOC estimates aligned with `iterates`. None values render as "--".
    iterate_label : str, optional
        Header text for the iterate column. Default "x_n". Use "m_n"
        for bisection so the column header reflects "midpoint".
    include_ratio : bool, optional
        If True, an additional column displaying e_{k+1}/e_k is
        printed. This is required for the bisection test case (vii)
        in the PA2 spec. Default False.

    Returns
    -------
    None
        This function only prints to standard output; it does not
        return a value.
    """

    # ---- Variable declarations ----
    n = len(iterates)              # number of iterates to print (integer)
    ratio = 0.0                    # successive-error ratio e_{k+1}/e_k (double float)
    ratio_str = ""                 # formatted ratio string for printing (str)
    eoc_str = ""                   # formatted EOC string for printing  (str)

    # ---- Print the header row ----
    if include_ratio:
        print(f"{'k':>4s}  {iterate_label:>22s}  "
              f"{'|x_n - r|':>15s}  {'p_k (EOC)':>12s}  "
              f"{'e_{k+1}/e_k':>14s}")
        print("-" * 76)
    else:
        print(f"{'k':>4s}  {iterate_label:>22s}  "
              f"{'|x_n - r|':>15s}  {'p_k (EOC)':>12s}")
        print("-" * 60)

    # ---- Print one row per iterate ----
    # Loop over each index k. For each row we print: k, x_k, e_k,
    # the EOC estimate (if available), and optionally e_{k+1}/e_k.
    for k in range(n):

        # EOC: render None as "--" so the column stays visually aligned.
        if eocs[k] is None:
            eoc_str = "--"
        else:
            eoc_str = f"{eocs[k]:.6f}"

        # Successive-error ratio: only meaningful when we have a next
        # error AND the current error is strictly positive (otherwise
        # the ratio is 0/0 or undefined).
        if include_ratio:
            if k + 1 < n and errors[k] > 0.0 and errors[k + 1] >= 0.0:
                ratio = errors[k + 1] / errors[k]
                ratio_str = f"{ratio:.6f}"
            else:
                ratio_str = "--"

            print(f"{k:4d}  {iterates[k]:22.15f}  "
                  f"{errors[k]:15.6e}  {eoc_str:>12s}  "
                  f"{ratio_str:>14s}")
        else:
            print(f"{k:4d}  {iterates[k]:22.15f}  "
                  f"{errors[k]:15.6e}  {eoc_str:>12s}")


# =============================================================================
# RUNNER FUNCTIONS
# -----------------------------------------------------------------------------
# Thin wrappers that handle the boilerplate around running one method
# on one test case: print a header, call the method, compute errors and
# EOCs, print the table, and print a summary footer.
#
# These wrappers also catch exceptions (e.g., the ValueError raised by
# bisection_method when its sign-change hypothesis fails) so that an
# expected-failure test case can still print a clean, informative
# message instead of crashing the whole script.
# =============================================================================


def run_newton_test(label, f, f_prime, x0, true_root,
                    tol=1e-14, max_iter=100, expected_behavior=""):
    """
    Run Newton's method on a single test case and print results.

    Parameters
    ----------
    label : str
        Short name for the test case, printed at the top of the table.
    f, f_prime : callables
        Function and its derivative.
    x0 : float
        Initial guess.
    true_root : float
        Known true root r used to compute absolute errors.
    tol : float, optional
        Convergence tolerance passed to newtons_method.
    max_iter : int, optional
        Iteration cap passed to newtons_method.
    expected_behavior : str, optional
        Free-form text describing what theory predicts. Printed in
        the header so the table can be read alongside expectations.

    Returns
    -------
    None
    """
    print("=" * 76)
    print(f"NEWTON'S METHOD  -- {label}")
    print(f"  x0 = {x0},  true root r = {true_root}")
    if expected_behavior:
        print(f"  Expected: {expected_behavior}")
    print("=" * 76)

    try:
        root, num_iter, iterates, residuals = newtons_method(
            f, f_prime, x0,
            tol=tol, max_iter=max_iter, return_iterates=True
        )
    except ZeroDivisionError as err:
        print(f"  Newton's method raised ZeroDivisionError: {err}")
        print("=" * 76)
        print()
        return

    errors = absolute_errors(iterates, true_root)
    eocs = experimental_order(errors)
    print_convergence_table(iterates, errors, eocs, iterate_label="x_n")

    print("-" * 76)
    print(f"  Final iterate:    {root:.15f}")
    print(f"  Iterations used:  {num_iter}")
    print("=" * 76)
    print()


def run_bisection_test(label, f, a, b, true_root,
                       tol=1e-14, max_iter=200,
                       include_ratio=False, expected_behavior=""):
    """
    Run Bisection on a single test case and print results.

    Parameters
    ----------
    label : str
        Short name for the test case.
    f : callable
        Function whose root is sought.
    a, b : float
        Interval endpoints passed to bisection_method.
    true_root : float
        Known true root used to compute errors.
    tol, max_iter : optional
        Forwarded to bisection_method.
    include_ratio : bool, optional
        If True, the printed table includes the e_{k+1}/e_k column
        required for test case (vii). Defaults to False.
    expected_behavior : str, optional
        Theory commentary printed in the header.

    Returns
    -------
    None
    """
    print("=" * 76)
    print(f"BISECTION METHOD -- {label}")
    print(f"  [a, b] = [{a}, {b}],  true root r = {true_root}")
    if expected_behavior:
        print(f"  Expected: {expected_behavior}")
    print("=" * 76)

    try:
        root, num_iter, iterates, residuals = bisection_method(
            f, a, b,
            tol=tol, max_iter=max_iter, return_iterates=True
        )
    except ValueError as err:
        # Bisection raises ValueError when f(a)*f(b) >= 0. This is the
        # expected outcome for test case (iii), so we report it cleanly
        # instead of letting the script crash.
        print(f"  Bisection rejected the input: {err}")
        print("=" * 76)
        print()
        return

    errors = absolute_errors(iterates, true_root)
    eocs = experimental_order(errors)
    print_convergence_table(iterates, errors, eocs,
                            iterate_label="m_n",
                            include_ratio=include_ratio)

    # If we are running the bisection-EOC test (vii), additionally
    # report the average successive-error ratio for the iterations
    # where both errors are strictly positive.
    if include_ratio:
        valid_ratios = []          # list of valid e_{k+1}/e_k values (list of float)
        for k in range(len(errors) - 1):
            if errors[k] > 0.0 and errors[k + 1] >= 0.0:
                valid_ratios.append(errors[k + 1] / errors[k])
        if valid_ratios:
            mean_ratio = sum(valid_ratios) / len(valid_ratios)
            print("-" * 76)
            print(f"  Average successive-error ratio e_(k+1)/e_k: {mean_ratio:.6f}")
            print(f"  (Theory predicts a value bounded above by 1, often near 1/2.)")

    print("-" * 76)
    print(f"  Final midpoint:   {root:.15f}")
    print(f"  Iterations used:  {num_iter}")
    print("=" * 76)
    print()


def run_secant_test(label, f, x0, x1, true_root,
                    tol=1e-14, max_iter=100, expected_behavior=""):
    """
    Run Secant Method on a single test case and print results.

    Parameters
    ----------
    label : str
        Short name for the test case.
    f : callable
        Function whose root is sought.
    x0, x1 : float
    true_root : float
        Known true root used to compute errors.
    tol, max_iter : optional
        Forwarded to secant_method.
    expected_behavior : str, optional
        Theory commentary printed in the header.

    Returns
    -------
    None
    """
    print("=" * 76)
    print(f"SECANT METHOD    -- {label}")
    print(f"  x0 = {x0}, x1 = {x1},  true root r = {true_root}")
    if expected_behavior:
        print(f"  Expected: {expected_behavior}")
    print("=" * 76)

    try:
        root, num_iter, iterates, residuals = secant_method(
            f, x0, x1,
            tol=tol, max_iter=max_iter, return_iterates=True
        )
    except ZeroDivisionError as err:
        print(f"  Secant method raised ZeroDivisionError: {err}")
        print("=" * 76)
        print()
        return

    errors = absolute_errors(iterates, true_root)
    eocs = experimental_order(errors)
    print_convergence_table(iterates, errors, eocs, iterate_label="x_n")

    print("-" * 76)
    print(f"  Final iterate:    {root:.15f}")
    print(f"  Iterations used:  {num_iter}")
    print("=" * 76)
    print()


# =============================================================================
# LAYER 2: TEST CASES (i) - (vii)
# -----------------------------------------------------------------------------
# Each function below corresponds to one of the seven required test
# items in the PA2 Part 2 spec. They are intentionally left as STUBS
# for now; we will fill them in one at a time, after confirming
# theoretical justification for each choice of f, initial guesses, etc.
#
# Each stub explains what the test must demonstrate so that the choice
# of function and starting conditions can be discussed before coding.
# =============================================================================


def test_case_i_simple_root_A():
    """
    PA2 item (i), example A:
    f(x) = x^2 - 2, true root r = sqrt(2) ~ 1.41421356...

    f'(r) = 2*sqrt(2) != 0, so all three methods satisfy their
    theoretical requirements for convergence to a simple root.
    Bisection on [1, 2]: f(1) = -1 < 0, f(2) = 2 > 0 -- sign change confirmed.
    """

    f       = lambda x: x**2 - 2       # define f(x) = x^2 - 2
    f_prime = lambda x: 2*x            # define f'(x) = 2x (analytical derivative)
    r       = math.sqrt(2)             # true root known exactly as sqrt(2)

    run_newton_test(
        label="(i.A) f(x)=x^2-2, r=sqrt(2)",
        f=f, f_prime=f_prime,
        x0=1.0,                        # initial guess close to sqrt(2)
        true_root=r,
        expected_behavior="Quadratic convergence; all methods succeed."
    )
    run_bisection_test(
        label="(i.A) f(x)=x^2-2, r=sqrt(2)",
        f=f,
        a=1.0, b=2.0,                  # [1,2] brackets the root: f(1)<0, f(2)>0
        true_root=r,
        expected_behavior="Linear convergence; sign change on [1,2] confirmed."
    )
    run_secant_test(
        label="(i.A) f(x)=x^2-2, r=sqrt(2)",
        f=f,
        x0=1.0, x1=2.0,               # two initial guesses bracketing the root
        true_root=r,
        expected_behavior="Superlinear convergence at order ~1.618."
    )


def test_case_i_simple_root_B():
    """
    PA2 item (i), example B:
    f(x) = cos(x) - x, true root r = Dottie number ~ 0.7390851332...

    f'(x) = -sin(x) - 1.  At r: f'(r) = -sin(r) - 1 ~ -1.674 != 0.
    So all three convergence theorems apply.
    Bisection on [0, 1]: f(0) = 1 > 0, f(1) = cos(1)-1 ~ -0.46 < 0 -- sign change confirmed.
    True root obtained via Newton to tol=1e-15 and cross-checked with scipy.
    """

    f       = lambda x: math.cos(x) - x        # define f(x) = cos(x) - x
    f_prime = lambda x: -math.sin(x) - 1       # define f'(x) = -sin(x) - 1

    # Since sqrt(Dottie number) has no closed form, we compute a
    # high-accuracy reference root using Newton at very tight tolerance.
    # Quadratic convergence makes this accurate to machine precision.
    r, _, _, _ = newtons_method(f, f_prime, x0=0.7, tol=1e-15, max_iter=100,
                                return_iterates=True)  # reference root ~ 0.7390851332

    run_newton_test(
        label="(i.B) f(x)=cos(x)-x, r~0.73909",
        f=f, f_prime=f_prime,
        x0=0.5,                        # initial guess near the root
        true_root=r,
        expected_behavior="Quadratic convergence; all methods succeed."
    )
    run_bisection_test(
        label="(i.B) f(x)=cos(x)-x, r~0.73909",
        f=f,
        a=0.0, b=1.0,                  # [0,1] brackets the root: f(0)>0, f(1)<0
        true_root=r,
        expected_behavior="Linear convergence; sign change on [0,1] confirmed."
    )
    run_secant_test(
        label="(i.B) f(x)=cos(x)-x, r~0.73909",
        f=f,
        x0=0.5, x1=1.0,               # two initial guesses on either side of root
        true_root=r,
        expected_behavior="Superlinear convergence at order ~1.618."
    )


def test_case_ii_newton_failure():
    """
    PA2 item (ii):
    f(x) = x^2, true root r = 0.  This is a double root: f'(r) = 0.

    Newton's theorem for quadratic convergence requires f'(r) != 0.
    When f'(r) = 0, the method still converges but only linearly.
    Specifically, the iteration becomes x_{n+1} = x_n - x_n^2/(2*x_n) = x_n/2,
    so each step halves the error -- linear convergence at rate 1/2,
    NOT the quadratic rate the method normally achieves.
    The EOC column should stabilize near 1.0, not 2.0.
    """

    f       = lambda x: x**2           # f(x) = x^2, double root at x = 0
    f_prime = lambda x: 2*x            # f'(x) = 2x, which equals 0 at the root
    r       = 0.0                      # true root is exactly 0

    run_newton_test(
        label="(ii) f(x)=x^2 double root, f'(r)=0",
        f=f, f_prime=f_prime,
        x0=1.0,                        # start away from root to observe slow convergence
        true_root=r,
        expected_behavior="LINEAR convergence (order~1) because f'(r)=0 -- NOT quadratic."
    )


def test_case_iii_invalid_bisection():
    """
    PA2 item (iii):
    f(x) = x^2 + 1 on [-2, 2].

    f has no real roots (x^2 + 1 >= 1 for all real x).
    f(-2) = 5 > 0 and f(2) = 5 > 0, so f(a)*f(b) = 25 > 0.
    The sign-change hypothesis f(a)*f(b) < 0 is violated.
    bisection_method should raise a ValueError -- that is the
    expected and correct behavior here.
    """

    f = lambda x: x**2 + 1            # f(x) = x^2 + 1, no real roots exist

    # run_bisection_test catches the ValueError and prints it cleanly.
    # true_root=0.0 is a placeholder only; the method will error before using it.
    run_bisection_test(
        label="(iii) f(x)=x^2+1, no real roots on [-2,2]",
        f=f,
        a=-2.0, b=2.0,                 # f(-2)=5 and f(2)=5, both positive -- no sign change
        true_root=0.0,                 # placeholder, never reached
        expected_behavior="ValueError expected: f(a)*f(b) > 0, sign-change hypothesis fails."
    )


def test_case_iv_secant_failure():
    """
    PA2 item (iv):
    f(x) = x^2, true root r = 0.  Double root, f'(r) = 0.

    The theoretical order (1+sqrt(5))/2 for Secant Method requires
    the same conditions as Newton's method: f'(r) != 0 and f''(r) != 0.
    When f'(r) = 0 (double root), the denominator in the secant step
    shrinks much slower than expected and the method degrades to
    linear convergence, well below the golden-ratio order of 1.618.
    The EOC column should stabilize near 1.0, not 1.618.
    """

    f = lambda x: x**2                 # f(x) = x^2, double root at x = 0
    r = 0.0                            # true root is exactly 0

    run_secant_test(
        label="(iv) f(x)=x^2 double root, f'(r)=0",
        f=f,
        x0=1.0, x1=0.5,               # two starting guesses away from the root
        true_root=r,
        expected_behavior="Slower than order 1.618 because f'(r)=0 violates convergence hypothesis."
    )


def test_case_v_secant_golden_ratio():
    """
    PA2 item (v):
    f(x) = x^2 - 2, true root r = sqrt(2).

    Conditions for golden-ratio convergence of Secant Method
    (per PA2 spec: the theoretical requirements for Newton's method
    must hold):
       * f'(r) = 2*sqrt(2) != 0   -- simple root
       * f''(r) = 2 != 0          -- nonzero second derivative
       * x0, x1 sufficiently close to r
    All conditions are satisfied.  EOC should stabilize near
    (1+sqrt(5))/2 ~ 1.6180339887...
    """

    f = lambda x: x**2 - 2            # f(x) = x^2 - 2, simple root at sqrt(2)
    r = math.sqrt(2)                   # true root known exactly

    run_secant_test(
        label="(v) f(x)=x^2-2, Secant order~1.618",
        f=f,
        x0=1.0, x1=1.5,               # starting guesses close enough to the root
        true_root=r,
        expected_behavior=f"Superlinear convergence at order ~ {(1+math.sqrt(5))/2:.6f}."
    )


def test_case_vi_newton_quadratic():
    """
    PA2 item (vi):
    f(x) = x^2 - 2, true root r = sqrt(2).

    Newton's theorem: if f'(r) != 0 and x0 is sufficiently close
    to r, then |e_{n+1}| <= C * e_n^2 where
        C = |f''(r)| / (2 * |f'(r)|) = 2 / (2 * 2*sqrt(2)) = 1/(2*sqrt(2)) ~ 0.354.
    Here f'(r) = 2*sqrt(2) != 0 and f''(r) = 2 != 0, so quadratic
    convergence is guaranteed.  The EOC column should stabilize near 2.0.
    """

    f       = lambda x: x**2 - 2      # f(x) = x^2 - 2, simple root at sqrt(2)
    f_prime = lambda x: 2*x           # f'(x) = 2x, nonzero at the root
    r       = math.sqrt(2)            # true root known exactly

    run_newton_test(
        label="(vi) f(x)=x^2-2, Newton order~2",
        f=f, f_prime=f_prime,
        x0=1.0,                        # initial guess close to sqrt(2)
        true_root=r,
        expected_behavior="Quadratic convergence (order~2); C = 1/(2*sqrt(2)) ~ 0.354."
    )


def test_case_vii_bisection_linear():
    """
    PA2 item (vii):
    f(x) = x^2 - 2, true root r = sqrt(2), interval [1, 2].

    Bisection is linearly convergent: after n steps the error
    satisfies e_n <= (b-a) / 2^n = 1 / 2^n.
    Successive errors satisfy e_{k+1} / e_k <= 1/2 in the ideal case,
    so the ratio column should hover near 0.5 and the EOC near 1.0.
    include_ratio=True triggers the extra column and average ratio
    summary required by the PA2 spec for this test case.
    """

    f = lambda x: x**2 - 2            # f(x) = x^2 - 2, simple root at sqrt(2)
    r = math.sqrt(2)                   # true root known exactly

    run_bisection_test(
        label="(vii) f(x)=x^2-2, Bisection order~1",
        f=f,
        a=1.0, b=2.0,                  # [1,2] brackets the root: f(1)<0, f(2)>0
        true_root=r,
        include_ratio=True,            # adds e_{k+1}/e_k column, required by PA2 spec
        expected_behavior="Linear convergence (order~1); e_{k+1}/e_k ~ 0.5 each step."
    )

# =============================================================================
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
# Running this file from the command line executes every test case in
# order. Once a test case is implemented, it will print its full
# convergence table here.
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 76)
    print("# PA2 PART 2 -- TESTING HARNESS")
    print("# All seven required test cases (i) through (vii)")
    print("#" * 76 + "\n")

    test_case_i_simple_root_A()
    test_case_i_simple_root_B()
    test_case_ii_newton_failure()
    test_case_iii_invalid_bisection()
    test_case_iv_secant_failure()
    test_case_v_secant_golden_ratio()
    test_case_vi_newton_quadratic()
    test_case_vii_bisection_linear()

    print("#" * 76)
    print("# END OF PART 2 TESTING HARNESS")
    print("#" * 76)

 
