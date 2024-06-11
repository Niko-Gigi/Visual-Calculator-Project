import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
NUM_PARTICLES = 1000
DT = 0.01  # Time step
FO = 0.998  # Friction coefficient
DP = 0.009  # Drag coefficient
CX = 0.00235  # X component of the force field
CY = 0.00005  # Y component of the force field

# Function to calculate position and velocity based on position
def position_based_function(position):
    velocity = position * 0.9  # Velocity depends linearly on position
    return velocity

# Generate initial positions for particles
positions = np.random.rand(NUM_PARTICLES, 2) * 10 - 5  # Random positions in the range [-5, 5]

# Calculate initial velocities based on positions
velocities = position_based_function(positions)

# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Create a scatter plot for particles
scatter = ax.scatter(positions[:, 0], positions[:, 1], s=2)

# Update function for animation
def update(frame):
    global positions, velocities

    # Compute the force field
    force_field = np.array([CX, CY])

    # Update velocities based on the force field, friction, and drag
    velocities += force_field * DT
    velocities *= FO
    velocities -= velocities * DP

    # Update positions based on velocities
    positions += velocities * DT

    # Apply periodic boundary conditions
    positions = np.where(positions > 5, positions - 10, positions)
    positions = np.where(positions < -5, positions + 10, positions)

    # Update the scatter plot
    scatter.set_offsets(positions)

    return scatter,

# Create animation
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# Show the animation
plt.show()
