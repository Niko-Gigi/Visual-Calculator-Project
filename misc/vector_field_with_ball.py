import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a grid of points for plotting the vector field
x = np.linspace(-10, 10, 21)
y = np.linspace(-10, 10, 21)
X, Y = np.meshgrid(x, y)

# Create a figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Define the vector field equations (these can be changed)
def U_func(X, Y):
    return -X*X - Y

def V_func(X, Y):
    return X - Y

# Define the vector field based on the U and V components
def vector_field(X, Y):
    U = U_func(X, Y)
    V = V_func(X, Y)
    return U, V

# Generate the vector field values over the grid
U, V = vector_field(X, Y)
# Create an initial quiver plot (vector field plot)
quiver = ax.quiver(X, Y, U, V)

# Use sympy to calculate the divergence of the vector field
x_sym, y_sym = sym.symbols('x y', real=True)
U_sym = U_func(x_sym, y_sym)
V_sym = V_func(x_sym, y_sym)

# Calculate the partial derivatives needed for divergence
dU_dx_sym = sym.diff(U_sym, x_sym)
dV_dy_sym = sym.diff(V_sym, y_sym)
divergence_sym = dU_dx_sym + dV_dy_sym

# Create a lambda function for numerical evaluation of divergence
divergence_func = sym.lambdify((x_sym, y_sym), divergence_sym, 'numpy')

# Evaluate divergence over the grid of points
divergence = divergence_func(X, Y)

# Animation parameters
dt = 0.1  # Time step for animation
drag = 0.1  # Drag force coefficient
# Initial conditions for the ball (position and velocity)
X_0, Y_0, Vx_0, Vy_0 = 0, 3, 0, 0
ball, = ax.plot(X_0, Y_0, 'ro')  # Plot the initial position of the ball

# Update function for the ball's animation
def update_ball(frame):
    global X_0, Y_0, Vx_0, Vy_0
    U_0, V_0 = vector_field(X_0, Y_0)  # Calculate vector field at the ball's position
    accx, accy = U_0 - drag * Vx_0, V_0 - drag * Vy_0  # Calculate acceleration

    # Update velocity based on acceleration
    Vx_0 += accx * dt
    Vy_0 += accy * dt

    # Update position based on velocity
    X_0 += Vx_0 * dt
    Y_0 += Vy_0 * dt

    # Update the ball's position on the plot
    ball.set_data([X_0], [Y_0])  # Wrap X_0 and Y_0 in lists
    return ball,

# Create the animation
anim = FuncAnimation(fig, update_ball, frames=100, interval=100)

# Annotation setup for hover interaction
annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                    textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# Hover function to display vector field values and divergence
def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        xdata, ydata = event.xdata, event.ydata  # Get the mouse position
        distances = np.hypot(X - xdata, Y - ydata)  # Calculate distances to all grid points
        i, j = np.unravel_index(np.argmin(distances), distances.shape)  # Find the closest point

        if distances[i, j] < 0.5:  # If the mouse is close enough to a point
            x_pos, y_pos = X[i, j], Y[i, j]  # Get the grid point position
            u, v = U[i, j], V[i, j]  # Get the vector components at the grid point
            div = divergence[i, j]  # Get the divergence at the grid point
            annot.xy = (x_pos, y_pos)
            text = f"({x_pos:.2f}, {y_pos:.2f})\nU={u:.2f}, V={v:.2f}\nDiv={div:.2f}"  # Create annotation text
            annot.set_text(text)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

# Connect the hover function to the motion_notify_event
fig.canvas.mpl_connect("motion_notify_event", hover)

# Show the plot
plt.show()