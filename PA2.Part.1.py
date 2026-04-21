###############################################################################
# Program: PA2.Part.1.py
# Author(s): Griffin Fee, Jack Zettlemoyer, Kai Behrens, Jacob Fitzmaurice
# Date: 4/21/26
# Purpose: This program implements classical root-finding methods for
#          Math 248, Programming Assignment 2. Currently implemented:
#          Newton's Method. Additional methods (Bisection, Secant) can
#          be added following the same function signature conventions.
#
#          To use, import this file and call the desired root-finding
#          function with the appropriate inputs. See individual function
#          docstrings for detailed usage instructions.
###############################################################################
# Inputs:  See individual function docstrings below.
# Outputs: See individual function docstrings below.
###############################################################################

# =============================================================================


def newtons_method(f, f_prime, x0, tol = 1e-10, max_iter = 100, return_iterates = False):
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

    Examples
    --------
    >>> # Find the root of f(x) = x^2 - 2 (i.e., sqrt(2))
    >>> f = lambda x: x**2 - 2
    >>> f_prime = lambda x: 2*x
    >>> root, iters = newtons_method(f, f_prime, 1.5)
    >>> print(root)
    1.4142135623730951
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
        iterates_list = [x_current]         # stores every x_n
        residuals_list = [abs(f(x_current))] # stores |f(x_n)| at each step

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
        iterates_list = []                   # stores every midpoint m
        residuals_list = []                  # stores |f(m)| at each step

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


# =============================================================================

def secant_method(f, x0, x1, tol = 1e-10, max_iter = 100, return_iterates = False):
    """
    Find a root of f(x) = 0 using the secant method.

    The secant method approximates the derivative in Newton's method with a
    finite difference, using the two most recent iterates to construct a
    secant line whose x-intercept becomes the next iterate:

        x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    Unlike Newton's method, no derivative evaluation is required, making
    this suitable for functions where f'(x) is expensive or unavailable.
    The method converges linearly with order approximately 1.618
    (the golden ratio) near a simple root.

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
    x_previous = float(x0)             # previous iterate x_{n-1} (double float)
    x_next = 0.0                   # next iterate x_{n+1} (double float)
    f_val_curr = 0.0                    # f(x_n) at current iterate (double float)
    f_val_prev = 0.0               # f(x_{n-1}) at previous iterate (double float)
    diff = 0.0                     # |x_{n+1} - x_n| (double float)
    iteration_count = 0            # number of iterations performed (integer)

    # If the user wants the full history, initialize storage lists
    if return_iterates:
        iterates_list = [x_previous, x_current]         # stores every x_n
        residuals_list = [abs(f(x_previous)), abs(f(x_current))] # stores |f(x_n)| at each step

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
        # x_{n + 1} = x_n - f(x_n) * ( (x_n - x_{n-1}) / ( f(x_n) - f(x_{n-1}) )
        x_next = x_current - f_val_curr * ( (x_current - x_previous) / (f_val_curr - f_val_prev))

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


# =============================================================================
# Quick verification (can be removed or moved to a test script later)
# =============================================================================
if __name__ == "__main__":

    # ================================================================
    # Quick verification: find sqrt(2) as the root of f(x) = x^2 - 2
    #   Known root: r = 1.41421356237...
    #   f'(x) = 2x
    # ================================================================

    f_test = lambda x: x**2 - 2
    f_prime_test = lambda x: 2 * x

    # ----------------------------------------------------------------
    # Test 1: Newton's Method (x0 = 1.5)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Newton's Method Test: f(x) = x^2 - 2, x0 = 1.5")
    print("Expected root: 1.41421356237...")
    print("=" * 60)

    root, num_iter, iterates, residuals = newtons_method(
        f_test, f_prime_test, 1.5, return_iterates=True
    )

    # Display results in a table format
    print(f"{'Iter':>4s}  {'x_n':>20s}  {'|f(x_n)|':>15s}")
    print("-" * 45)
    for i in range(len(iterates)):
        print(f"{i:4d}  {iterates[i]:20.15f}  {residuals[i]:15.10e}")

    print("-" * 45)
    print(f"Approximate root: {root:.15f}")
    print(f"Iterations used:  {num_iter}")
    print(f"|f(root)|:        {abs(f_test(root)):.5e}")
    print("=" * 60)

    print()  # blank line between tests

    # ----------------------------------------------------------------
    # Test 2: Bisection Method (a = 0, b = 2)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Bisection Method Test: f(x) = x^2 - 2, [a, b] = [0, 2]")
    print("Expected root: 1.41421356237...")
    print("=" * 60)

    root, num_iter, iterates, residuals = bisection_method(
        f_test, 0, 2, return_iterates=True
    )

    # Display results in a table format
    print(f"{'Iter':>4s}  {'m_n':>20s}  {'|f(m_n)|':>15s}")
    print("-" * 45)
    for i in range(len(iterates)):
        print(f"{i+1:4d}  {iterates[i]:20.15f}  {residuals[i]:15.10e}")

    print("-" * 45)
    print(f"Approximate root: {root:.15f}")
    print(f"Iterations used:  {num_iter}")
    print(f"|f(root)|:        {abs(f_test(root)):.5e}")
    print("=" * 60)

    print()  # blank line between tests

    # ----------------------------------------------------------------
    # Test 3: Secant Method (x0 = 1, x1 = 2)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Secant Method Test: f(x) = x^2 - 2, x0 = 1, x1 = 2")
    print("Expected root: 1.41421356237...")
    print("=" * 60)

    root, num_iter, iterates, residuals = secant_method(
        f_test, 1, 2, return_iterates=True
    )

    # Display results in a table format
    print(f"{'Iter':>4s}  {'x_n':>20s}  {'|f(x_n)|':>15s}")
    print("-" * 45)
    for i in range(len(iterates)):
        print(f"{i:4d}  {iterates[i]:20.15f}  {residuals[i]:15.10e}")

    print("-" * 45)
    print(f"Approximate root: {root:.15f}")
    print(f"Iterations used:  {num_iter}")
    print(f"|f(root)|:        {abs(f_test(root)):.5e}")
    print("=" * 60)