import numpy as np
from sympy import symbols, sympify, lambdify
import math

def bisection_method(equation_str, a, b, iter=10):
    """
    Find a root of an equation using the bisection method.
    
    Args:
        equation_str (str): String representation of the equation
        a (float): Left endpoint of the interval
        b (float): Right endpoint of the interval
        iter (integer): Number of iterations done
    
    Returns:
        tuple: (root, plot_points)
            root (float or None): The found root or None if no root found
            plot_points (list): List of points for plotting
    """
    # Convert string equation to symbolic expression
    x = symbols('x')
    expr = sympify(equation_str)
    f = lambdify(x, expr)
    
    # Generate plot points
    x_values = np.linspace(a, b, 100)
    y_values = [float(f(x)) for x in x_values]
    plot_points = [{'x': float(x), 'y': float(y)} for x, y in zip(x_values, y_values)]
    
    # Check if a root exists in the interval
    if f(a) * f(b) >= 0:
        return None, plot_points
    
    # Bisection algorithm
    c = a
    for _ in range(iter):
        c = (a + b) / 2
        
        if f(c) == 0:
            return c, plot_points
        
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2, plot_points

def newtonraphson_method(equation_str, x0, iter=10):
    """
    Find a root of an equation using the Newton-Raphson method.
    
    Args:
        equation_str (str): String representation of the equation
        x0 (float): Initial guess
        iter (integer): Number of iterations
    
    Returns:
        tuple: (root, plot_points, iterations)
            root (float or None): The found root or None if no convergence
            plot_points (list): List of points for plotting
            iterations (list): List of x values at each iteration
    """
    # Convert string equation to symbolic expression
    x = symbols('x')
    expr = sympify(equation_str)
    
    # Create function and its derivative
    f = lambdify(x, expr)
    f_prime = lambdify(x, math.diff(expr, x))
    
    # Generate plot points
    x_range = x0 * 2 if x0 != 0 else 2  # Determine a reasonable plot range
    x_values = np.linspace(x0 - x_range, x0 + x_range, 100)
    y_values = [float(f(x)) for x in x_values]
    plot_points = [{'x': float(x), 'y': float(y)} for x, y in zip(x_values, y_values)]
    
    # Newton-Raphson algorithm
    x_n = x0
    iterations = [x_n]
    
    try:
        for _ in range(iter):
            f_x = f(x_n)
            f_prime_x = f_prime(x_n)
            
            # Check if derivative is close to zero
            if abs(f_prime_x) < 1e-10:
                return None, plot_points, iterations
            
            # Newton-Raphson formula
            x_n_plus_1 = x_n - f_x / f_prime_x
            
            # Check for convergence
            if abs(x_n_plus_1 - x_n) < 1e-10:
                return x_n_plus_1, plot_points, iterations
            
            x_n = x_n_plus_1
            iterations.append(x_n)
    
    except (ZeroDivisionError, OverflowError):
        return None, plot_points, iterations
    
    return x_n, plot_points, iterations
