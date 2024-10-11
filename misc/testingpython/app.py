from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from io import BytesIO
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    x, y = sym.symbols('x y')
    
    # Parse the user inputs for U and V
    U_expr = data.get('U')
    V_expr = data.get('V')
    
    # Convert to sympy expressions
    U = sym.lambdify((x, y), U_expr)
    V = sym.lambdify((x, y), V_expr)
    
    # Calculate curl (only z-component in 2D)
    curl_z = sym.diff(V_expr, x) - sym.diff(U_expr, y)
    
    # Calculate divergence
    divergence = sym.diff(U_expr, x) + sym.diff(V_expr, y)
    
    # Lambdify curl and divergence for numerical evaluation
    curl_z_func = sym.lambdify((x, y), curl_z)
    divergence_func = sym.lambdify((x, y), divergence)
    
    return jsonify({
        'curl_z': str(curl_z),
        'divergence': str(divergence),
        'curl_z_value': curl_z_func(data.get('pointX'), data.get('pointY')),
        'divergence_value': divergence_func(data.get('pointX'), data.get('pointY'))
    })

@app.route('/vector-field', methods=['POST'])
def vector_field():
    data = request.json
    U_expr = data.get('U')
    V_expr = data.get('V')
    
    # Define vector field functions based on user input
    U = sym.lambdify(('x', 'y'), U_expr, modules='numpy')
    V = sym.lambdify(('x', 'y'), V_expr, modules='numpy')

    def create_vector_field_plot(Fx, Fy):
        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        X, Y = np.meshgrid(x, y)
        U = Fx(X, Y)
        V = Fy(X, Y)
        
        fig, ax = plt.subplots()
        ax.quiver(X, Y, U, V)
        ax.set_title("Vector Field")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        return buf

    buf = create_vector_field_plot(U, V)
    return send_file(buf, mimetype='image/png')

@app.route('/particle-animation', methods=['POST'])
def particle_animation():
    data = request.json
    U_expr = data.get('U')
    V_expr = data.get('V')
    
    # Define vector field functions based on user input
    U = sym.lambdify(('x', 'y'), U_expr, modules='numpy')
    V = sym.lambdify(('x', 'y'), V_expr, modules='numpy')

    def create_particle_animation(Fx, Fy, duration=3, num_particles=700, particle_lifetime=100, tail_length=20):
        fig, ax = plt.subplots()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
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

        # Save animation to a temporary file
        temp_filename = 'temp_animation.gif'
        ani.save(temp_filename, writer='pillow', fps=30)

        # Read the saved file into a BytesIO object
        buf = BytesIO()
        with open(temp_filename, 'rb') as f:
            buf.write(f.read())
        buf.seek(0)

        # Clean up the temporary file
        os.remove(temp_filename)
        
        return buf

    buf = create_particle_animation(U, V)
    return send_file(buf, mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)




