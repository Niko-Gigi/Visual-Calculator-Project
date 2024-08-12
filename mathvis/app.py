from flask import Flask, render_template, send_file, request
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from matplotlib.animation import FuncAnimation
import io
import math
from scipy.optimize import fsolve, minimize
from scipy.integrate import solve_ivp
import base64
from PIL import Image, ImageDraw

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vector_calculus')
def vector_calculus():
    # Create vector field plot
    x = np.linspace(-10, 10, 21)
    y = np.linspace(-10, 10, 21)
    X, Y = np.meshgrid(x, y)

    def U_func(X, Y):
        return -X*X - Y

    def V_func(X, Y):
        return X - Y

    def vector_field(X, Y):
        U = U_func(X, Y)
        V = V_func(X, Y)
        return U, V

    U, V = vector_field(X, Y)

    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    quiver = ax.quiver(X, Y, U, V)

    x_sym, y_sym = sym.symbols('x y', real=True)
    U_sym = U_func(x_sym, y_sym)
    V_sym = V_func(x_sym, y_sym)

    dU_dx_sym = sym.diff(U_sym, x_sym)
    dV_dy_sym = sym.diff(V_sym, y_sym)
    divergence_sym = dU_dx_sym + dV_dy_sym

    divergence_func = sym.lambdify((x_sym, y_sym), divergence_sym, 'numpy')
    divergence = divergence_func(X, Y)

    dt = 0.1
    drag = 0.1
    X_0, Y_0, Vx_0, Vy_0 = 0, 3, 0, 0
    ball, = ax.plot(X_0, Y_0, 'ro')

    def update_ball(frame):
        nonlocal X_0, Y_0, Vx_0, Vy_0
        U_0, V_0 = vector_field(X_0, Y_0)
        accx, accy = U_0 - drag * Vx_0, V_0 - drag * Vy_0
        Vx_0 += accx * dt
        Vy_0 += accy * dt
        X_0 += Vx_0 * dt
        Y_0 += Vy_0 * dt
        ball.set_data([X_0], [Y_0])
        return ball,

    anim = FuncAnimation(fig, update_ball, frames=100, interval=100)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            xdata, ydata = event.xdata, event.ydata
            distances = np.hypot(X - xdata, Y - ydata)
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            if distances[i, j] < 0.5:
                x_pos, y_pos = X[i, j], Y[i, j]
                u, v = U[i, j], V[i, j]
                div = divergence[i, j]
                annot.xy = (x_pos, y_pos)
                text = f"({x_pos:.2f}, {y_pos:.2f})\nU={u:.2f}, V={v:.2f}\nDiv={div:.2f}"
                annot.set_text(text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    # Save plot to a BytesIO object and return as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/numerical_methods', methods=['GET', 'POST'])
def numerical_methods():
    if request.method == 'POST':
        method = request.form['method']
        func = request.form['function']
        iterations = int(request.form['iterations'])
        lower_bound = float(request.form['lower_bound'])
        upper_bound = float(request.form['upper_bound'])
        guess_1 = float(request.form['guess_1'])
        guess_2 = float(request.form.get('guess_2', 0))
        x = sym.symbols('x')
        fx = sym.sympify(func)

        if method == 'newton':
            img = newton_raphson(fx, iterations, x, guess_1)
        elif method == 'secant':
            img = secant(fx, iterations, x, guess_1, guess_2)
        elif method == 'bisection':
            img = bisection(fx, iterations, x, lower_bound, upper_bound, guess_1, guess_2)

        return render_template('numerical_methods.html', plot_url=img)

    return render_template('numerical_methods.html')

def newton_raphson(fx, iterations, x, xi):
    fx_der = sym.diff(fx, x)
    img = plot_iterations(x, fx, fx_der, xi, iterations, 'Newton-Raphson')
    return img

def secant(fx, iterations, x, x_first, x_second):
    img = plot_iterations(x, fx, None, x_first, iterations, 'Secant', x_second=x_second)
    return img

def bisection(fx, iterations, x, xl, xu, guess_1, guess_2):
    img = plot_iterations(x, fx, None, guess_1, iterations, 'Bisection', xl=xl, xu=xu)
    return img

def plot_iterations(x, fx, fx_der, xi, iterations, method_name, x_second=None, xl=None, xu=None):
    frames = []
    x_vals = np.linspace(-10, 10, 400)
    y_vals = [fx.evalf(subs={x: val}) for val in x_vals]

    for i in range(min(iterations, 120)):
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label='f(x)')
        
        if method_name == 'Newton-Raphson':
            tangent = fx_der.evalf(subs={x: xi})
            xi_old = xi
            xi = xi - fx.evalf(subs={x: xi}) / tangent
            plt.plot([xi_old, xi], [fx.evalf(subs={x: xi_old}), 0], 'ro-')
        elif method_name == 'Secant':
            tmp = x_second - fx.evalf(subs={x: x_second}) * (xi - x_second) / (fx.evalf(subs={x: xi}) - fx.evalf(subs={x: x_second}))
            plt.plot([xi, tmp], [fx.evalf(subs={x: xi}), 0], 'go-')
            xi = x_second
            x_second = tmp
        elif method_name == 'Bisection':
            xi_old = xi
            xi = (xl + xu) / 2
            plt.plot([xi_old, xi], [fx.evalf(subs={x: xi_old}), 0], 'bo-')
            if fx.evalf(subs={x: xi}) * fx.evalf(subs={x: xl}) < 0:
                xu = xi
            else:
                xl = xi

        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title(f'{method_name} Method Iteration {i+1}')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()

    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format='GIF', save_all=True, append_images=frames[1:], duration=500, loop=0)
    gif_buf.seek(0)
    img = base64.b64encode(gif_buf.getvalue()).decode('utf8')

    return f'data:image/gif;base64,{img}'

@app.route('/linear_algebra')
def linear_algebra():
    # Example linear algebra visualization
    def plot_matrix_transformation(matrix):
        fig, ax = plt.subplots()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(True)

        # Define the original vectors
        origin = np.zeros(2)
        vectors = np.array([[1, 0], [0, 1]])

        # Plot the original vectors
        ax.quiver(*origin, *vectors.T, color=['r', 'b'], scale=1, scale_units='xy', angles='xy', width=0.005)
        
        # Apply the matrix transformation
        transformed_vectors = matrix @ vectors.T

        # Plot the transformed vectors
        ax.quiver(*origin, *transformed_vectors, color=['r', 'b'], scale=1, scale_units='xy', angles='xy', width=0.005, linestyle='--')

        return fig

    matrix = np.array([[2, 1], [1, 2]])
    fig = plot_matrix_transformation(matrix)

    # Save plot to a BytesIO object and return as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/ode')
def ode():
    return render_template('ode.html')

@app.route('/cobweb_diagram')
def cobweb_diagram():
    # Cobweb Diagram Code
    def f(x):
        return 4.5*x*(1-x)

    def f_prime(x):
        return 4.5 - 9*x

    class CobwebDiagram:
        def __init__(self, f, f_prime, initial_guess, iterations):
            self.f = f
            self.f_prime = f_prime
            self.initial_guess = initial_guess
            self.iterations = iterations
            self.x_min, self.x_max, self.y_min, self.y_max = self.calculate_bounds()

        def calculate_bounds(self):
            zeros = fsolve(self.f, [-1, 0, 1])
            result = minimize(lambda x: -self.f(x), 0)
            max_val = -result.fun

            x_min = min(zeros) - 0.25
            x_max = max(zeros) + 0.25
            y_min = min(self.f(np.array(zeros)).tolist() + [max_val]) - 0.25
            y_max = max(self.f(np.array(zeros)).tolist() + [max_val]) + 0.25

            return x_min, x_max, y_min, y_max

        def plot(self):
            x = np.linspace(self.x_min, self.x_max, 400)
            y = self.f(x)
            
            fig, ax = plt.subplots()
            ax.plot(x, y, label='$f(x)$')
            ax.plot(x, x, 'r--', label='$y=x$')

            x_vals = [self.initial_guess]
            y_vals = [self.f(self.initial_guess)]

            ax.plot(self.initial_guess, self.f(self.initial_guess), 'go')  

            for _ in range(self.iterations):
                x_next = self.f(x_vals[-1])
                y_vals.append(x_next)
                x_vals.append(x_next)

                ax.plot([x_vals[-2], x_vals[-2]], [y_vals[-2], y_vals[-1]], 'b-')
                ax.plot([x_vals[-2], x_vals[-1]], [y_vals[-1], y_vals[-1]], 'b-')

            ax.plot(x_vals[:-1], y_vals[:-1], 'b-', label='Cobweb Path')

            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.legend()
            plt.title('Cobweb Diagram')
            plt.grid()
            plt.savefig('static/cobweb_diagram.png')
            plt.close()

    cobweb = CobwebDiagram(
        f=f,
        f_prime=f_prime,
        initial_guess=0.234,
        iterations=30
    )

    cobweb.plot()
    return send_file('static/cobweb_diagram.png', mimetype='image/png')

@app.route('/phase_portrait')
def phase_portrait():
    # Phase Portrait Code
    def system(t, state):
        x, y = state
        dxdt = y
        dydt = -x + y 
        return [dxdt, dydt]

    class PhasePortrait:
        def __init__(self, system, x_range=None, y_range=None, grid_density=20):
            self.system = system
            self.grid_density = grid_density
            self.x_range, self.y_range = self.calculate_bounds(x_range, y_range)

        def calculate_bounds(self, x_range, y_range):
            if x_range is None or y_range is None:
                x_samples = np.linspace(-3, 3, 100)
                y_samples = np.linspace(-3, 3, 100)
                X, Y = np.meshgrid(x_samples, y_samples)
                U, V = np.zeros_like(X), np.zeros_like(Y)

                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        dxdt, dydt = self.system(0, [X[i, j], Y[i, j]])
                        U[i, j] = dxdt
                        V[i, j] = dydt

                U_flat = U.flatten()
                V_flat = V.flatten()
                x_min, x_max = np.min(X), np.max(X)
                y_min, y_max = np.min(Y), np.max(Y)

                x_margin = (x_max - x_min) * 0.2
                y_margin = (y_max - y_min) * 0.2

                return (x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin)
            else:
                return x_range, y_range

        def vector_field(self, x, y):
            dxdt, dydt = self.system(0, [x, y])
            return dxdt, dydt

        def plot(self):
            x = np.linspace(self.x_range[0], self.x_range[1], self.grid_density)
            y = np.linspace(self.y_range[0], self.y_range[1], self.grid_density)
            X, Y = np.meshgrid(x, y)
            U, V = np.zeros_like(X), np.zeros_like(Y)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    dxdt, dydt = self.vector_field(X[i, j], Y[i, j])
                    U[i, j] = dxdt
                    V[i, j] = dydt

            plt.figure(figsize=(10, 8))
            plt.quiver(X, Y, U, V, color='r', headlength=3)

            for x0 in np.linspace(self.x_range[0], self.x_range[1], 5):
                for y0 in np.linspace(self.y_range[0], self.y_range[1], 5):
                    sol = solve_ivp(self.system, [0, 10], [x0, y0], t_eval=np.linspace(0, 10, 500))
                    plt.plot(sol.y[0], sol.y[1], 'b-', alpha=0.7)

            plt.xlim(self.x_range)
            plt.ylim(self.y_range)
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.title('Phase Portrait')
            plt.grid()
            plt.savefig('static/phase_portrait.png')
            plt.close()

    portrait = PhasePortrait(
        system=system,
        x_range=None,
        y_range=None
    )

    portrait.plot()
    return send_file('static/phase_portrait.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)


