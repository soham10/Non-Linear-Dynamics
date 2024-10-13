import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import welch

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
t = np.linspace(0, num_steps*dt, num_steps)

# Initial conditions
X = np.array([1.0, 0.0, 0.0])
Y = np.array([1.1, 0.0, 0.0])

# Arrays to store the time series data
time_series1 = np.empty((num_steps, 3))
time_series2 = np.empty((num_steps, 3))

# Generate the time series data
for i in range(num_steps):
    X = rk4_step(rossler, X, t[i], dt, a, b, c)
    time_series1[i] = X

for i in range(num_steps):
    Y = rk4_step(rossler, Y, t[i], dt, a, b, c)
    time_series2[i] = Y

# Plotting the time series
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, time_series1[:, 0], color='b')
plt.plot(t, time_series2[:, 0], color='r')
plt.title('Time Series of x')
plt.xlabel('Time')
plt.ylabel('x')

plt.subplot(3, 1, 2)
plt.plot(t, time_series1[:, 1], color='b')
plt.plot(t, time_series2[:, 1], color='r')
plt.title('Time Series of y')
plt.xlabel('Time')
plt.ylabel('y')

plt.subplot(3, 1, 3)
plt.plot(t, time_series1[:, 2], color='b')
plt.plot(t, time_series2[:, 2], color='r')
plt.title('Time Series of z')
plt.xlabel('Time')
plt.ylabel('z')

plt.tight_layout()
plt.show()