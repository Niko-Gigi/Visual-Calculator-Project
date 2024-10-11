import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox

# Define a function to create a vector field plot
def plot_vector_field(x_func, y_func):
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    U = x_func(X)
    V = y_func(Y)

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, scale=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Vector Field')
    plt.grid()
    plt.show()

# Define functions for x and y components
def_x_func = 'x'  # Default x component function
def_y_func = 'y'  # Default y component function

# Create text boxes for user input of x and y functions
x_func_text_box = TextBox(plt.axes([0.1, 0.01, 0.8, 0.05]), 'x Function:', initial=def_x_func)
y_func_text_box = TextBox(plt.axes([0.1, 0.07, 0.8, 0.05]), 'y Function:', initial=def_y_func)

# Function to update the plot when text is entered into the text boxes
def update_plot(text):
    x_func = lambda x: eval(x)
    y_func = lambda y: eval(y)
    plot_vector_field(x_func, y_func)

x_func_text_box.on_submit(update_plot)
y_func_text_box.on_submit(update_plot)

plt.show()
