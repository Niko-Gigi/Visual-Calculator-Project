# Basic vector field animation code quickly made with ChatGPT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a grid of points
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y) # a matrix of numbers

# Define the vector field lol
def vector_field(X, Y):
    U = -Y
    V = 2*X
    return U, V # U and V are the x and y components of the vector arrows

# Create a figure and axis (for pyplot)
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Create an initial quiver plot
U, V = vector_field(X, Y)
quiver = ax.quiver(X, Y, U, V) # creates an arrow with x component U and y component V with its tail at position X, Y


# Function to update the quiver plot
def update_quiver(num, quiver, X, Y):
    # Update the vector field to animate
    U, V = vector_field(X * np.cos(num / 10) - Y * np.sin(num / 10), X * np.sin(num / 10) + Y * np.cos(num / 10))
    quiver.set_UVC(U, V)
    return quiver,

# Create an animation
anim = FuncAnimation(fig, update_quiver, fargs=(quiver, X, Y), frames=100, interval=100)


plt.show()