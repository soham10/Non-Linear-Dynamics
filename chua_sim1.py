import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for Chua's circuit
alpha = 15.6
beta = 33.8
m0 = -1.143
m1 = -0.714

# Define the function f(x)
def f(x):
    return m1 * x + (m0 - m1) / 2 * (np.abs(x + 1) - np.abs(x - 1))

# Define the differential equations
def chua_derivatives(state, t):
    x, y, z = state
    dxdt = alpha * (y - x - f(x))
    dydt = x - y + z
    dzdt = -beta * y
    return np.array([dxdt, dydt, dzdt])

# Implement RK4 method for ODEs
def rk4_step(derivatives, state, t, dt):
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Initial conditions
x0, y0, z0 = 0.1, 0.0, 0.0  # Initial state
state = np.array([x0, y0, z0])

# Time parameters
t0 = 0.0   # Initial time
t_end = 150.0  # End time
dt = 0.001  # Time step
times = np.arange(t0, t_end, dt)
for beta in np.arange(36,33.2,-0.01):
    # Arrays to store the solution
    x_values = []
    y_values = []
    z_values = []

    # Perform the RK4 simulation
    for t in times:
        x_values.append(state[0])
        y_values.append(state[1])
        z_values.append(state[2])
        state = rk4_step(chua_derivatives, state, t, dt)

    x_values_p = x_values[int(100/dt):]
    y_values_p = y_values[int(100/dt):]
    z_values_p = z_values[int(100/dt):]

    arr = []

    for i in range(1,len(x_values_p)-1):
        if(x_values_p[i-1]<x_values_p[i] and x_values_p[i]>x_values_p[i+1]):
            arr.append(x_values_p[i])

    print(beta)

    plt.scatter([beta]*len(arr),arr, s=0.1, c = 'b')
plt.gca().invert_xaxis()

plt.show()