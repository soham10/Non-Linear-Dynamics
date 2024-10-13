import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters for the Rössler attractor
a, b, c = 0.2, 0.2, 5.7

# Define the Rössler system
def rossler(X, t, a, b, c):
    x, y, z = X
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return np.array([dxdt, dydt, dzdt])

# Implementing the RK4 method
def rk4_step(f, X, t, dt, *args):
    k1 = dt * f(X, t, *args)
    k2 = dt * f(X + 0.5 * k1, t + 0.5 * dt, *args)
    k3 = dt * f(X + 0.5 * k2, t + 0.5 * dt, *args)
    k4 = dt * f(X + k3, t + dt, *args)
    return X + (k1 + 2*k2 + 2*k3 + k4) / 6

# Initialize variables
dt = 0.01  # Time step
num_steps = 10000  # Number of steps
t = 0

# Initial conditions for the two bodies
X1 = np.array([1.0, 0.0, 0.0])
X2 = np.array([1.1, 0.0, 0.0])

# Arrays to store the trajectories
trajectory1 = np.empty((num_steps, 3))
trajectory2 = np.empty((num_steps, 3))

# Generate the trajectories
for i in range(num_steps):
    X1 = rk4_step(rossler, X1, t, dt, a, b, c)
    X2 = rk4_step(rossler, X2, t, dt, a, b, c)
    trajectory1[i] = X1
    trajectory2[i] = X2
    t += dt

# Plotting and Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(0, 25)

line1, = ax.plot([], [], [], lw=2, color='b', label='Body 1')
line2, = ax.plot([], [], [], lw=2, color='r', label='Body 2')

def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    return line1, line2

def update(frame):
    line1.set_data(trajectory1[:frame, 0], trajectory1[:frame, 1])
    line1.set_3d_properties(trajectory1[:frame, 2])
    line2.set_data(trajectory2[:frame, 0], trajectory2[:frame, 1])
    line2.set_3d_properties(trajectory2[:frame, 2])
    return line1, line2

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=1)

ax.legend()
plt.show()
