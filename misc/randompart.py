import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constants
NUM_PARTICLES = 100
DT = 0.01  # Time step
CX = 0.00235  # X component of the force field
CY = 0.00005  # Y component of the force field
CZ = 0.001  # Z component of the force field
PARTICLE_RADIUS = 0.05  # Radius of particles

# Function to calculate position and velocity based on position
def position_based_function(position):
    velocity = position * 0.9  # Velocity depends linearly on position
    return velocity

# Generate initial positions for particles
positions = np.random.rand(NUM_PARTICLES, 3) * 10 - 5  # Random positions in the range [-5, 5]

# Calculate initial velocities based on positions
velocities = position_based_function(positions)

# Create the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

# Create a scatter plot for particles
scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=2)

# Display kinetic energy and temperature on screen
energy_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

# Function to handle particle collisions
def handle_collisions(positions, velocities):
    for i in range(NUM_PARTICLES):
        for j in range(i + 1, NUM_PARTICLES):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2 * PARTICLE_RADIUS:
                # Compute the new velocities after the collision
                vel_i = velocities[i]
                vel_j = velocities[j]
                pos_i = positions[i]
                pos_j = positions[j]

                # Compute the normal and tangential components of the velocities
                normal = (pos_i - pos_j) / dist
                tangent = np.cross(normal, np.array([0, 0, 1]))
                tangent /= np.linalg.norm(tangent)

                # Decompose velocities into normal and tangential components
                v_i_normal = np.dot(vel_i, normal)
                v_i_tangent = np.dot(vel_i, tangent)
                v_j_normal = np.dot(vel_j, normal)
                v_j_tangent = np.dot(vel_j, tangent)

                # Swap the normal components for elastic collision
                new_v_i_normal = v_j_normal
                new_v_j_normal = v_i_normal

                # Recompose the velocities
                velocities[i] = new_v_i_normal * normal + v_i_tangent * tangent
                velocities[j] = new_v_j_normal * normal + v_j_tangent * tangent

    return velocities

# Function to calculate kinetic energy
def calculate_kinetic_energy(velocities):
    kinetic_energy = 0.5 * np.sum(velocities**2)
    return kinetic_energy

# Update function for animation
def update(frame):
    global positions, velocities

    # Compute the force field
    force_field = np.array([CX, CY, CZ])

    # Update velocities based on the force field
    velocities += force_field * DT

    # Handle collisions
    velocities = handle_collisions(positions, velocities)

    # Update positions based on velocities
    positions += velocities * DT

    # Handle collisions with borders
    for i in range(NUM_PARTICLES):
        if positions[i, 0] < ax.get_xlim()[0] + PARTICLE_RADIUS or positions[i, 0] > ax.get_xlim()[1] - PARTICLE_RADIUS:
            velocities[i, 0] *= -1
        if positions[i, 1] < ax.get_ylim()[0] + PARTICLE_RADIUS or positions[i, 1] > ax.get_ylim()[1] - PARTICLE_RADIUS:
            velocities[i, 1] *= -1
        if positions[i, 2] < ax.get_zlim()[0] + PARTICLE_RADIUS or positions[i, 2] > ax.get_zlim()[1] - PARTICLE_RADIUS:
            velocities[i, 2] *= -1

    # Update the scatter plot
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

    # Calculate and display the kinetic energy and temperature
    kinetic_energy = calculate_kinetic_energy(velocities)
    temperature = kinetic_energy / NUM_PARTICLES  # Assuming temperature is proportional to average kinetic energy
    energy_text.set_text(f"Kinetic Energy: {kinetic_energy:.2f}, Temperature: {temperature:.2f}")

    return scatter, energy_text

# Zoom function to adjust the volume
def on_scroll(event):
    scale_factor = 1.1 if event.button == 'up' else 1 / 1.1
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    current_zlim = ax.get_zlim()
    new_xlim = (current_xlim[0] * scale_factor, current_xlim[1] * scale_factor)
    new_ylim = (current_ylim[0] * scale_factor, current_ylim[1] * scale_factor)
    new_zlim = (current_zlim[0] * scale_factor, current_zlim[1] * scale_factor)
    
    # Calculate the center of the plot
    center_x = (current_xlim[0] + current_xlim[1]) / 2
    center_y = (current_ylim[0] + current_ylim[1]) / 2
    center_z = (current_zlim[0] + current_zlim[1]) / 2

    # Scale positions relative to the center
    positions[:, 0] = center_x + (positions[:, 0] - center_x) * scale_factor
    positions[:, 1] = center_y + (positions[:, 1] - center_y) * scale_factor
    positions[:, 2] = center_z + (positions[:, 2] - center_z) * scale_factor

    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)
    ax.set_zlim(new_zlim)
    fig.canvas.draw_idle()

# Connect the scroll event to the zoom function
fig.canvas.mpl_connect('scroll_event', on_scroll)

# Create animation
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# Show the animation
plt.show()

