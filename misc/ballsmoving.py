import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example vector field function
def Fx(x, y):
    return -np.sin(x) + np.exp(-y)

def Fy(x, y):
    return x + np.cos(y)

# Plot the vector field and animate particles with tails
def vector_field_plot(Fx, Fy, duration=3, num_particles=700, particle_lifetime=100, tail_length=20):
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 10)
    X, Y = np.meshgrid(x, y)
    U = Fx(X, Y)
    V = Fy(X, Y)
    
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V)
    
    # Initialize particles at random positions
    particles = np.random.rand(num_particles, 2) * 10 - 5  # random positions in the range [-5, 5]
    particle_lifetimes = np.random.randint(0, particle_lifetime, num_particles)  # random lifetimes
    
    # Initialize history to store tails, start with NaNs
    history = np.full((num_particles, tail_length, 2), np.nan)
    
    particle_paths = [ax.plot([], [], '-', lw=1)[0] for _ in range(num_particles)]  # lines for tails
    particle_heads, = ax.plot(particles[:, 0], particles[:, 1], 'o', markersize=2, color='blue')  # smaller heads

    def update(frame):
        nonlocal particles, particle_lifetimes, history
        
        # Update particle positions based on the vector field
        U = Fx(particles[:, 0], particles[:, 1])
        V = Fy(particles[:, 0], particles[:, 1])
        particles[:, 0] += U * 0.1
        particles[:, 1] += V * 0.1
        
        # Reduce lifetime for each particle
        particle_lifetimes -= 1
        
        # Respawn particles that have exceeded their lifetime
        for i in range(num_particles):
            if particle_lifetimes[i] <= 0:
                particles[i] = np.random.rand(2) * 10 - 5  # new random position
                particle_lifetimes[i] = particle_lifetime  # reset lifetime
                history[i] = np.nan  # reset history to NaNs to avoid tails from origin
        
        # Update history for tails
        history = np.roll(history, -1, axis=1)
        history[:, -1, :] = particles
        
        # Set particle tails
        for i, path in enumerate(particle_paths):
            path.set_data(history[i, :, 0], history[i, :, 1])
        
        # Set particle heads
        particle_heads.set_data(particles[:, 0], particles[:, 1])
        
        return particle_paths + [particle_heads]

    ani = animation.FuncAnimation(fig, update, frames=int(duration * 30), blit=True, interval=33)
    plt.show()

# Usage
Fx_input = Fx
Fy_input = Fy
vector_field_plot(Fx_input, Fy_input)