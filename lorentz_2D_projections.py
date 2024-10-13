import numpy as np
import matplotlib.pyplot as plt

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

# Create 2D projections
fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor='k')

# x-y projection
axs[0].plot(evol[:, 0], evol[:, 1], color='lime', lw=0.9)
axs[0].set_title('Projection onto x-y Plane', color='white')
axs[0].set_xlabel('X', color='white')
axs[0].set_ylabel('Y', color='white')
axs[0].set_facecolor('k')
axs[0].tick_params(axis='both', colors='white')
axs[0].grid(True, color='gray', linestyle='--', linewidth=0.5)

# x-z projection
axs[1].plot(evol[:, 0], evol[:, 2], color='lime', lw=0.9)
axs[1].set_title('Projection onto x-z Plane', color='white')
axs[1].set_xlabel('X', color='white')
axs[1].set_ylabel('Z', color='white')
axs[1].set_facecolor('k')
axs[1].tick_params(axis='both', colors='white')
axs[1].grid(True, color='gray', linestyle='--', linewidth=0.5)

# y-z projection
axs[2].plot(evol[:, 1], evol[:, 2], color='lime', lw=0.9)
axs[2].set_title('Projection onto y-z Plane', color='white')
axs[2].set_xlabel('Y', color='white')
axs[2].set_ylabel('Z', color='white')
axs[2].set_facecolor('k')
axs[2].tick_params(axis='both', colors='white')
axs[2].grid(True, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
