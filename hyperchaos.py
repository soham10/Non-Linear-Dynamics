import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the Folded-Towel map equations
def folded_towel_map(x, y, z):
    x_new = 3.8 * x * (1 - x) - 0.05 * (y + 0.35) * (1 - 2 * z)
    y_new = 0.1 * ((y + 0.35) * (1 - 2 * z) - 1) * (1 - 1.9 * x)
    z_new = 3.78 * z * (1 - z) + 0.2 * y
    return x_new, y_new, z_new

# Initialize the system with a number of iterations and initial conditions
n_points = 10000
x, y, z = np.zeros(n_points), np.zeros(n_points), np.zeros(n_points)
x[0], y[0], z[0] = 0.1, 0.1, 0.1  # Initial conditions

# Iterate the system
for i in range(1, n_points):
    x[i], y[i], z[i] = folded_towel_map(x[i-1], y[i-1], z[i-1])

# Create a 3D plot for the trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Setting up the plot limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Folded-Towel Map Attractor Animation')

# Creating a line object that will be updated during the animation
line, = ax.plot([], [], [], color='b', lw=0.5)

# Initialize function for the animation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

# Animation function that updates the plot
def update(num):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    return line,

# Creating the animation object
ani = FuncAnimation(fig, update, frames=n_points, init_func=init, blit=True, interval=1)

# To save the animation as a file, uncomment the line below (requires ffmpeg or imagemagick)
# ani.save('folded_towel_map_animation.mp4', writer='ffmpeg', fps=30)

# Display the animation
plt.show()
