import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __repr__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"

    def plot(self, ax, **kwargs):
        ax.quiver(0, 0, 0, self.x, self.y, self.z, **kwargs)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid()

# Example usage
if __name__ == "__main__":
    v1 = Vector3D(3, 4, 5)
    v2 = Vector3D(1, 2, 3)
    v3 = v1 + v2
    v4 = v1 - v2
    v5 = v1 * 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    v1.plot(ax, color='r', label='v1')
    v2.plot(ax, color='b', label='v2')
    v3.plot(ax, color='g', label='v1 + v2')
    v4.plot(ax, color='y', label='v1 - v2')
    v5.plot(ax, color='m', label='v1 * 2')
    plt.legend()
    plt.show()