import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define Chua's Circuit Parameters
alpha = 15.6
beta = 28.0
m0 = -1.143
m1 = -0.714

# Nonlinearity function g(x)
def g(x):
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

# Define the system of differential equations for Chua's circuit
def chua_system(state):
    x, y, z = state
    dx_dt = alpha * (y - x - g(x))
    dy_dt = x - y + z
    dz_dt = -beta * y
    return np.array([dx_dt, dy_dt, dz_dt])

# Runge-Kutta 4th Order Method
def rk4_step(func, state, dt):
    k1 = func(state)
    k2 = func(state + 0.5 * dt * k1)
    k3 = func(state + 0.5 * dt * k2)
    k4 = func(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Initialize the state of the system
initial_state = np.array([0.7, 0.0, 0.0])
dt = 0.01
num_steps = 10000

# Preallocate memory for the trajectory
trajectory = np.zeros((num_steps, 3))
trajectory[0] = initial_state

# Simulate the system
for i in range(1, num_steps):
    trajectory[i] = rk4_step(chua_system, trajectory[i-1], dt)

# Set up the figure and 3D axis
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

# Plot the initial point
line, = ax.plot([], [], [], lw=0.5)

# Update function for the animation
def update(num):
    line.set_data(trajectory[:num, 0], trajectory[:num, 1])
    line.set_3d_properties(trajectory[:num, 2])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=num_steps, interval=1, blit=True)

plt.show()