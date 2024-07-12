# Basic vector field animation code quickly made with ChatGPT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a grid of points
x = np.linspace(-10, 10, 21)
y = np.linspace(-10, 10, 21)
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

# Create the animation
anim = FuncAnimation(fig, update_ball, frames=100, interval=100)

# I'm experimenting with labels. I tried using mplcursors, but it was slow and complex. This approach below seems to work pretty well

# Annotate the point at position xy with label at position xy+xytext (if it didn't have textcoords="offset points" then label would be fixed at position xytext)
annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# A function which updates the label
def update_annot(i, j):
    # Next four lines give numerical values to x, y, u and v
    x_pos = X[i, j]
    y_pos = Y[i, j]
    u = U[i, j]
    v = V[i, j]
    # Then plug in these values for "" and xy into the annot variable from above
    annot.xy = (x_pos, y_pos)
    text = f"({x_pos:.2f}, {y_pos:.2f})\nU={u:.2f}, V={v:.2f}"
    annot.set_text(text)
    # Set box colour
    annot.get_bbox_patch().set_facecolor('white')
    # Set box opacity
    annot.get_bbox_patch().set_alpha(0.8)

# A function which defines what happens when the cursor hovers over a point
def hover(event):
    vis = annot.get_visible() # Checks whether the label is visible
    # If the cursor is within the plot's axes:
    if event.inaxes == ax:
        # The coordinates of the cursor
        xdata = event.xdata
        ydata = event.ydata
        
        distances = np.hypot(X - xdata, Y - ydata) # Takes the distances between the cursor and each point
        i,  j = np.unravel_index(np.argmin(distances), distances.shape) # Finds the smallest of these distances

        # If the cursor is close enough to one of the points, update the label
        if distances[i, j] < 0.5:
            update_annot(i, j)
            annot.set_visible(True)
            fig.canvas.draw_idle() # This redraws the canvas to show the updated label
        else:
            # If the cursor is not close to a point and the label is visible, hide it
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

# motion_notify_event is a predefined event in matplotlib which is triggered when the mouse is moved over the canvas
# mpl_connect calls the hover function and passes the event to it
fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()