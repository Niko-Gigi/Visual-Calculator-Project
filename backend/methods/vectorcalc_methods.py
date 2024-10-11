import numpy as np
from typing import Callable, Tuple

def divergence(F_x: Callable[[float, float], float], 
               F_y: Callable[[float, float], float], 
               x: float, y: float, h: float = 1e-6) -> float:
    """
    Calculate the divergence of a 2D vector field at point (x, y)
    
    Args:
        F_x: Function representing x-component of the vector field
        F_y: Function representing y-component of the vector field
        x: x-coordinate
        y: y-coordinate
        h: Step size for numerical differentiation
        
    Returns:
        Divergence value at point (x, y)
    """
    # Divergence = ∂F_x/∂x + ∂F_y/∂y
    dFx_dx = (F_x(x + h, y) - F_x(x - h, y)) / (2 * h)
    dFy_dy = (F_y(x, y + h) - F_y(x, y - h)) / (2 * h)
    
    return dFx_dx + dFy_dy

def curl(F_x: Callable[[float, float], float], 
         F_y: Callable[[float, float], float], 
         x: float, y: float, h: float = 1e-6) -> float:
    """
    Calculate the curl (z-component) of a 2D vector field at point (x, y)
    
    Args:
        F_x: Function representing x-component of the vector field
        F_y: Function representing y-component of the vector field
        x: x-coordinate
        y: y-coordinate
        h: Step size for numerical differentiation
        
    Returns:
        Curl (z-component) value at point (x, y)
    """
    # Curl in 2D = ∂F_y/∂x - ∂F_x/∂y (z-component only)
    dFy_dx = (F_y(x + h, y) - F_y(x - h, y)) / (2 * h)
    dFx_dy = (F_x(x, y + h) - F_x(x, y - h)) / (2 * h)
    
    return dFy_dx - dFx_dy