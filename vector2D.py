import matplotlib.pyplot as plt

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

    def plot(self, ax, **kwargs):
        ax.quiver(0, 0, self.x, self.y, angles='xy', scale_units='xy', scale=1, **kwargs)
        ax.grid()

def plot_vectors(vectors):
    fig, ax = plt.subplots()
    max_x = max([abs(vector.x) for vector in vectors])
    max_y = max([abs(vector.y) for vector in vectors])
    limit = max(max_x, max_y) * 1.1  # Adding 10% padding for better visualization

    for vector in vectors:
        vector.plot(ax)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    plt.show()

# Example usage
if __name__ == "__main__":
    v1 = Vector2D(0, 2)
    v2 = Vector2D(2, 0)
    v3 = Vector2D(-2, 0)
    v4 = Vector2D(0, -2)
    v5 = Vector2D(0, 2) - Vector2D(2, 2)

    vectors = [v1, v2, v3, v4, v5]
    plot_vectors(vectors)