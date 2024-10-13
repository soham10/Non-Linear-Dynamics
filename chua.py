import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Define Chua's Circuit Parameters
beta = 28.0
m0 = -1.143
m1 = -0.714
alpha_values = np.linspace(4, 10, 1000)  # Varying alpha from 4 to 10 with 1000 steps

# Nonlinearity function g(x)
def g(x):
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

# Define the system of differential equations for Chua's circuit
def chua_system(state, alpha):
    x, y, z = state
    dx_dt = alpha * (y - x - g(x))
    dy_dt = x - y + z
    dz_dt = -beta * y
    return np.array([dx_dt, dy_dt, dz_dt])

# Runge-Kutta 4th Order Method
def rk4_step(func, state, dt, alpha):
    k1 = func(state, alpha)
    k2 = func(state + 0.5 * dt * k1, alpha)
    k3 = func(state + 0.5 * dt * k2, alpha)
    k4 = func(state + dt * k3, alpha)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Bifurcation Diagram Data Collection
dt = 0.01
num_transient_steps = 5000  # Steps to remove transients
num_steady_state_steps = 500  # Steps to record steady states after transients
bifurcation_points = []

# Loop through varying alpha values
for alpha in alpha_values:
    # Initialize the state of the system
    state = np.array([0.7, 0.0, 0.0])
    
    # Skip the transient behavior
    for _ in range(num_transient_steps):
        state = rk4_step(chua_system, state, dt, alpha)
    
    # Collect steady-state behavior
    steady_states = []
    for _ in range(num_steady_state_steps):
        state = rk4_step(chua_system, state, dt, alpha)
        steady_states.append(state[0])  # Record x-coordinate
    
    # Find local maxima and minima in steady states
    steady_states = np.array(steady_states)
    local_maxima = argrelextrema(steady_states, np.greater)[0]  # Indices of local maxima
    local_minima = argrelextrema(steady_states, np.less)[0]     # Indices of local minima
    
    # Record the local maxima and minima for the bifurcation diagram
    bifurcation_points.extend([(alpha, steady_states[i]) for i in local_maxima])
    bifurcation_points.extend([(alpha, steady_states[i]) for i in local_minima])

# Extract alpha and x values for plotting
alpha_values_plot, x_values_plot = zip(*bifurcation_points)

# Plotting the Bifurcation Diagram (Local Maxima and Minima)
plt.figure(dpi=150)
plt.plot(alpha_values_plot, x_values_plot, ',k', alpha=0.5)  # ',' marker for dense plot
plt.title('Bifurcation Diagram of Chua\'s Circuit (Local Maxima and Minima)')
plt.xlabel(r'$\alpha$')
plt.ylabel('Local Maxima and Minima of x-values')
plt.show()
