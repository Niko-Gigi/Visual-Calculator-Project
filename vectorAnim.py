import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

class VectorFieldAnimation:
    def __init__(self, xlim, ylim, initial_grid_size):
        # Constructor method initializes the object
        self.xlim = xlim
        self.ylim = ylim
        self.grid_size = initial_grid_size
        
        # Create a new figure and axis for the plot
        self.fig, self.ax = plt.subplots()
        # Adjust the subplot to make room for widgets
        plt.subplots_adjust(left=0.25, bottom=0.25)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_aspect('equal')  # Set aspect ratio to equal
        
        # Create meshgrid for vector field
        self.X, self.Y = self.create_meshgrid(self.grid_size)
        # Initialize vector field and magnitudes
        self.U, self.V = self.vector_field1(self.X, self.Y, 0)
        self.magnitudes = np.sqrt(self.U**2 + self.V**2)
        # Normalize vectors
        self.U, self.V = self.normalize_vectors(self.U, self.V)
        # Create quiver plot
        self.quiver = self.ax.quiver(self.X, self.Y, self.U, self.V, scale=50, width=0.0025)

        # Initialize function and options
        self.vector_field_func = self.vector_field1
        self.use_colormap = False
        
        # Create UI widgets
        self.create_widgets()

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
        # Create a meshgrid for the vector field
        X, Y = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], grid_size),
                           np.linspace(self.ylim[0], self.ylim[1], grid_size))
        return X, Y

    def normalize_vectors(self, U, V):
        # Normalize the vectors
        magnitudes = np.sqrt(U**2 + V**2)
        return U / magnitudes, V / magnitudes

    # Define different vector field functions
    def vector_field1(self, X, Y, t):
        U = np.sin(X + t)
        V = np.cos(Y + t)
        return U, V

    def vector_field2(self, X, Y, t):
        U = X
        V = Y
        return U, V

    def vector_field3(self, X, Y, t):
        U = Y * np.cos(t)
        V = -X * np.sin(t)
        return U, V

    def update(self, frame):
        # Update function for animation
        t = frame / 10.0
        self.U, self.V = self.vector_field_func(self.X, self.Y, t)
        self.magnitudes = np.sqrt(self.U**2 + self.V**2)
        self.U, self.V = self.normalize_vectors(self.U, self.V)
        self.update_quiver()
        return self.quiver,

    def create_widgets(self):
        # Create slider for grid size
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Grid Size', 5, 30, valinit=self.grid_size, valstep=1)
        self.slider.on_changed(self.update_grid_size)
        
        # Create radio buttons for selecting vector field
        ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15], facecolor='lightgoldenrodyellow')
        self.radio = RadioButtons(ax_radio, ('Field 1', 'Field 2', 'Field 3'))
        self.radio.on_clicked(self.change_vector_field)

        # Create check buttons for colormap option
        ax_check = plt.axes([0.05, 0.2, 0.15, 0.15], facecolor='lightgoldenrodyellow')
        self.check = CheckButtons(ax_check, ['Use Colormap'], [self.use_colormap])
        self.check.on_clicked(self.toggle_colormap)

    def update_grid_size(self, val):
        # Update grid size based on slider value
        self.grid_size = int(self.slider.val)
        self.X, self.Y = self.create_meshgrid(self.grid_size)
        self.U, self.V = self.vector_field_func(self.X, self.Y, 0)
        self.magnitudes = np.sqrt(self.U**2 + self.V**2)
        self.U, self.V = self.normalize_vectors(self.U, self.V)
        self.update_quiver()
        self.fig.canvas.draw_idle()  # Refresh the plot

    def change_vector_field(self, label):
        # Change vector field based on radio button selection
        if label == 'Field 1':
            self.vector_field_func = self.vector_field1
        elif label == 'Field 2':
            self.vector_field_func = self.vector_field2
        elif label == 'Field 3':
            self.vector_field_func = self.vector_field3
        # Update the field immediately
        self.update_grid_size(None)

    def on_scroll(self, event):
        # Adjust the limits based on scroll direction
        scale_factor = 1.1
        if event.button == 'up':
            scale_factor = 1 / scale        
        self.xlim = (self.xlim[0] * scale_factor, self.xlim[1] * scale_factor)
        self.ylim = (self.ylim[0] * scale_factor, self.ylim[1] * scale_factor)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.X, self.Y = self.create_meshgrid(self.grid_size)
        self.U, self.V = self.vector_field_func(self.X, self.Y, 0)
        self.magnitudes = np.sqrt(self.U**2 + self.V**2)
        self.U, self.V = self.normalize_vectors(self.U, self.V)
        self.update_quiver()
        self.fig.canvas.draw_idle()  # Refresh the plot

    def on_press(self, event):
        # Handler for mouse press event
        if event.inaxes != self.ax:
            return
        self.press_event = (event.xdata, event.ydata)
        self.start_xlim = self.ax.get_xlim()
        self.start_ylim = self.ax.get_ylim()

    def on_release(self, event):
        # Handler for mouse release event
        self.press_event = None
        self.start_xlim = None
        self.start_ylim = None

    def on_motion(self, event):
        # Handler for mouse motion event
        if self.press_event is None or event.inaxes != self.ax:
            return
        dx = self.press_event[0] - event.xdata
        dy = self.press_event[1] - event.ydata
        self.ax.set_xlim(self.start_xlim[0] + dx, self.start_xlim[1] + dx)
        self.ax.set_ylim(self.start_ylim[0] + dy, self.start_ylim[1] + dy)
        self.fig.canvas.draw_idle()  # Refresh the plot

    def toggle_colormap(self, label):
        # Toggle colormap option
        self.use_colormap = not self.use_colormap
        self.update_quiver()
        self.fig.canvas.draw_idle()  # Refresh the plot

    def update_quiver(self):
        # Update quiver plot
        self.quiver.remove()  # Remove the old quiver
        if self.use_colormap:
            colors = self.magnitudes
            self.quiver = self.ax.quiver(self.X, self.Y, self.U, self.V, colors, scale=50, width=0.0025, cmap='coolwarm')
        else:
            self.quiver = self.ax.quiver(self.X, self.Y, self.U, self.V, scale=50, width=0.0025)

    def animate(self):
        # Function to animate the plot
        anim = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create an instance of VectorFieldAnimation
    vector_field_animation = VectorFieldAnimation(xlim=(-10, 10), ylim=(-10, 10), initial_grid_size=20)
    # Call the animate method to start the animation
    vector_field_animation.animate()
