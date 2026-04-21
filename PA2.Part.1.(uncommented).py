###############################################################################
# Program: PA2.Part.1.(uncommented).py
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


def newtons_method(f, f_prime, x0, tol = 1e-10, max_iter = 100, return_iterates = False):

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
        diff = abs(x_next - x_current)

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
# TODO: Add bisection_method() here following the same pattern.
#
#   def bisection_method(f, a, b, tol=1e-10, max_iter=100, return_iterates = False):
#       
#       
#
# =============================================================================


def secant_method(f, x0, x1, tol=1e-10, max_iter=100, return_iterates=False):
    
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
        diff = abs(x_next - x_current)

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

    # ---- Test: find sqrt(2) as the positive root of f(x) = x^2 - 2 ----
    #   Known root: r = 1.41421356237...
    #   f'(x) = 2x
    #   Initial guess: x0 = 1.5 (same as the textbook example)

    f_test = lambda x: x**2 - 2
    f_prime_test = lambda x: 2 * x
    x0_test = 1.5

    print("=" * 60)
    print("Newton's Method Test: f(x) = x^2 - 2, x0 = 1.5")
    print("Expected root: 1.41421356237...")
    print("=" * 60)

    # Run with iterates returned so we can inspect convergence
    root, num_iter, iterates, residuals = newtons_method(
        f_test, f_prime_test, x0_test, return_iterates=True
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