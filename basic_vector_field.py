# Basic vector field animation code quickly made with ChatGPT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a grid of points
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y) # a matrix of numbers

# Create a figure and axis (for pyplot)
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Define the vector field lol
def vector_field(X, Y):
    U = (-Y - X)
    V = (X - Y)
    return U, V # U and V are the x and y components of the vector arrows

# Create an initial quiver (vector) plot
U, V = vector_field(X, Y)
quiver = ax.quiver(X, Y, U, V) # creates an arrow with x component U and y component V with its tail at position X, Y

# Set timestep
dt = 0.1

# Set drag force parameter
drag = 0

# Create an initial test ball
X_0, Y_0, Vx_0, Vy_0 = 0, 3, 0, 0
ball, = ax.plot(X_0, Y_0, 'ro')

# Define function to update the ball position based on acceleration and velocity
def update_ball(frame):
    global X_0, Y_0, Vx_0, Vy_0

    accx, accy = vector_field(X_0, Y_0)[0]-drag*Vx_0, vector_field(X_0, Y_0)[1]-drag*Vy_0

    Vx_0 += accx*dt
    Vy_0 += accy*dt

    X_0 += Vx_0*dt
    Y_0 += Vy_0*dt

    ball.set_data(X_0, Y_0)

    trace, = ax.plot(X_0, Y_0, 'r.')

    return ball,

anim = FuncAnimation(fig, update_ball, frames=100, interval=100)

plt.show()