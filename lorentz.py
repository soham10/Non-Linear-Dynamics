import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Initial state
x, y, z = 0, 1, 10

# Parameters for the Lorenz system
rho, sigma, beta = 28, 10, 8/3

# Time parameters
t0 = 0
tf = 100
dt = 0.008
t = np.arange(t0, tf + dt, dt)  # Time array
n = len(t)

def lorenz_system(t, r):
    """ Defines the Lorenz system of ODEs. """
    x, y, z = r
    return np.array([
        sigma * (y - x),                # dx/dt
        rho * x - y - x * z,            # dy/dt
        x * y - beta * z                # dz/dt
    ])

def rk4(t, r, f, dt):
    """ Runge-Kutta 4th order integration. """
    k1 = dt * f(t, r)
    k2 = dt * f(t + dt / 2, r + k1 / 2)
    k3 = dt * f(t + dt / 2, r + k2 / 2)
    k4 = dt * f(t + dt, r + k3)
    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Initial condition
r = np.array([x, y, z])

# Array to store the evolution of the system
evol = np.zeros((n, 3))
evol[0] = r

# Compute the evolution using RK4 method
for i in range(n - 1):
    evol[i + 1] = rk4(t[i], evol[i], lorenz_system, dt)

# Create a 3D plot
fig = plt.figure('Lorenz Attractor', facecolor='k', figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    """ Update function for animation. """
    ax.clear()
    ax.set_facecolor('k')
    ax.set_axis_off()
    ax.plot(evol[:frame, 0], evol[:frame, 1], evol[:frame, 2], color='lime', lw=0.9)
    ax.view_init(elev=-6, azim=-56 + frame / 2)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=np.arange(1, n, 10), interval=20, repeat=False)

plt.show()
