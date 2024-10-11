import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, TextBox

class VectorField:
    def __init__(self, xlim, ylim, initial_grid_size, initial_particle_count):
        self.xlim = xlim
        self.ylim = ylim
        self.grid_size = initial_grid_size
        self.particle_count = initial_particle_count
        self.particle_lifetime = 100  # Number of frames a particle lives

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.35)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_aspect('equal')

        self.X, self.Y = self.create_meshgrid(self.grid_size)
        self.field_function_x = lambda X, Y: -np.cos(Y)  # Default function for x component
        self.field_function_y = lambda X, Y: np.sin(X)   # Default function for y component
        self.U, self.V = self.calculate_field(self.X, self.Y, self.field_function_x, self.field_function_y)
        self.quiver = self.ax.quiver(self.X, self.Y, self.U, self.V, scale=50, width=0.0025)

        self.create_widgets()

        # Particle initialization
        self.positions = np.random.rand(initial_particle_count, 2) * (xlim[1] - xlim[0]) + xlim[0]
        self.velocities = np.zeros((initial_particle_count, 2))
        self.ages = np.zeros(initial_particle_count)
        self.scatter = self.ax.scatter(self.positions[:, 0], self.positions[:, 1], s=2, c=self.ages, cmap='viridis', vmin=0, vmax=self.particle_lifetime)

        # Connect the scroll and mouse events to their handlers
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Variables to store the state of the pan
        self.press_event = None
        self.start_xlim = None
        self.start_ylim = None

    def create_meshgrid(self, grid_size):
        X, Y = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], grid_size),
                           np.linspace(self.ylim[0], self.ylim[1], grid_size))
        return X, Y

    def calculate_field(self, X, Y, field_function_x, field_function_y):
        U = field_function_x(X, Y)
        V = field_function_y(X, Y)
        magnitudes = np.sqrt(U**2 + V**2)
        U_normalized = U / magnitudes
        V_normalized = V / magnitudes
        return U_normalized, V_normalized

    def update(self, frame):
        # Update vector field
        self.U, self.V = self.calculate_field(self.X, self.Y, self.field_function_x, self.field_function_y)

        # Particle update
        grid_spacing_x = (self.xlim[1] - self.xlim[0]) / self.grid_size
        grid_spacing_y = (self.ylim[1] - self.ylim[0]) / self.grid_size
        grid_x = ((self.positions[:, 0] - self.xlim[0]) / grid_spacing_x).astype(int)
        grid_y = ((self.positions[:, 1] - self.ylim[0]) / grid_spacing_y).astype(int)
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)
        self.velocities[:, 0] = self.U[grid_y, grid_x]
        self.velocities[:, 1] = self.V[grid_y, grid_x]
        self.positions += self.velocities * 0.1  # Adjust the speed of the particles

        # Handle particle aging and respawn
        self.ages += 1
        respawn_indices = self.ages > self.particle_lifetime
        self.positions[respawn_indices] = np.random.rand(np.sum(respawn_indices), 2) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        self.ages[respawn_indices] = 0

        # Filter particles within bounds
        in_bounds = (self.positions[:, 0] >= self.xlim[0]) & (self.positions[:, 0] <= self.xlim[1]) & \
                    (self.positions[:, 1] >= self.ylim[0]) & (self.positions[:, 1] <= self.ylim[1])
        self.scatter.set_offsets(self.positions[in_bounds])
        self.scatter.set_array(self.ages[in_bounds])

        # Update the quiver plot
        self.quiver.set_UVC(self.U, self.V)
        return self.quiver, self.scatter

    def create_widgets(self):
        # Create slider for particle count
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Number of particles', 1, 100, valinit=self.particle_count, valstep=1)
        self.slider.on_changed(self.update_particle_number)

        # Create text input boxes for x and y component functions
        ax_input_x = plt.axes([0.025, 0.45, 0.2, 0.05])
        self.textbox_x = TextBox(ax_input_x, 'Function for x', initial=self.get_field_function_x_str())
        self.textbox_x.on_submit(self.update_field_function_x)

        ax_input_y = plt.axes([0.025, 0.35, 0.2, 0.05])
        self.textbox_y = TextBox(ax_input_y, 'Function for y', initial=self.get_field_function_y_str())
        self.textbox_y.on_submit(self.update_field_function_y)

    def update_particle_number(self, val):
        self.particle_count = int(val)
        # Add functionality to update particle count in your application

    def update_field_function_x(self, text):
        self.field_function_x = self.parse_field_function(text)
        self.update_grid_size(self.grid_size)

    def update_field_function_y(self, text):
        self.field_function_y = self.parse_field_function(text)
        self.update_grid_size(self.grid_size)

    def parse_field_function(self, text):
        allowed_names = {k: getattr(np, k) for k in dir(np) if not k.startswith("__")}
        allowed_names.update({'X': None, 'Y': None})  # Add X and Y as allowed names
        try:
            # Replace X and Y with dummy variables for validation
            eval(text, {"__builtins__": {}}, allowed_names)
            func = eval(f"lambda X, Y: {text}", {"__builtins__": {}}, allowed_names)
            return func
        except Exception as e:
            print(f"Error: {e}. Please enter a valid mathematical expression.")
            return None


    def get_field_function_x_str(self):
        return self.get_field_function_str(self.field_function_x)

    def get_field_function_y_str(self):
        return self.get_field_function_str(self.field_function_y)

    def get_field_function_str(self, func):
        return str(func) if func else ""

    def update_grid_size(self, val):
        self.grid_size = int(self.slider.val)
        self.X, self.Y = self.create_meshgrid(self.grid_size)
        self.U, self.V = self.calculate_field(self.X, self.Y, self.field_function_x, self.field_function_y)
        self.quiver.set_UVC(self.U, self.V)
        self.fig.canvas.draw_idle()  # Refresh the plot

    def update_particle_number(self, val):
        self.particle_count = int(self.slider.val)
        self.fig.canvas.draw_idle()  # Refresh the plot

    def on_scroll(self, event):
        # Adjust the limits based on scroll direction
        scale_factor = 1.1
        if event.button == 'up':
            scale_factor = 1 / scale_factor
        self.xlim = (self.xlim[0] * scale_factor, self.xlim[1] * scale_factor)
        self.ylim = (self.ylim[0] * scale_factor, self.ylim[1] * scale_factor)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.X, self.Y = self.create_meshgrid(self.grid_size)
        self.U, self.V = self.calculate_field(self.X, self.Y, self.field_function_x, self.field_function_y)
        self.quiver.set_UVC(self.U, self.V)
        self.fig.canvas.draw_idle()  # Refresh the plot

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.press_event = (event.xdata, event.ydata)
        self.start_xlim = self.ax.get_xlim()
        self.start_ylim = self.ax.get_ylim()

    def on_release(self, event):
        self.press_event = None
        self.start_xlim = None
        self.start_ylim = None

    def on_motion(self, event):
        if self.press_event is None or event.inaxes != self.ax:
            return
        dx = self.press_event[0] - event.xdata
        dy = self.press_event[1] - event.ydata
        self.ax.set_xlim(self.start_xlim[0] + dx, self.start_xlim[1] + dx)
        self.ax.set_ylim(self.start_ylim[0] + dy, self.start_ylim[1] + dy)
        self.update_grid_size(self.grid_size)  # Update the vectors to appear correctly
        self.fig.canvas.draw_idle()  # Refresh the plot

    def animate(self):
        anim = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=True)
        plt.show()

# Example usage
if __name__ == "__main__":
    vector_field_animation = VectorField(xlim=(-10, 10), ylim=(-10, 10), initial_grid_size=20, initial_particle_count=2000)
    vector_field_animation.animate()


