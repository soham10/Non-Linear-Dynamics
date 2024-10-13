import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
omega_0 = 2.0  # Natural angular frequency of the oscillator
omega = 3.0    # Driving angular frequency
F = 1.0        # Amplitude of the driving force
m = 1.0        # Mass of the oscillator (set to 1 for simplicity)
t_fixed = 5.0  # Fixed time to evaluate the displacement
x0 = 0.0       # Initial displacement
v0 = 0.0       # Initial velocity
gammas = np.linspace(0, 10, 500)  # Varying gamma from 0 to 10

# Differential equation for the forced damped harmonic oscillator
def forced_damped_oscillator(t, y, gamma, omega, F, omega_0, m):
    x, v = y
    dxdt = v
    dvdt = (F/m) * np.sin(omega * t) - 2 * gamma * v - omega_0**2 * x
    return [dxdt, dvdt]

# Array to store displacement values for different gammas
displacements = []

for gamma in gammas:
    # Initial conditions
    y0 = [x0, v0]
    
    # Solve the differential equation
    sol = solve_ivp(forced_damped_oscillator, [0, t_fixed], y0, args=(gamma, omega, F, omega_0, m), t_eval=[t_fixed])
    
    # Store the displacement at time t_fixed
    displacements.append(sol.y[0][0])

# Plotting x vs gamma
plt.figure(figsize=(10, 6))
plt.plot(gammas, displacements, label=f'x(t={t_fixed}) vs gamma', color='b')

plt.title('Displacement vs Damping Coefficient for Forced Harmonic Oscillator')
plt.xlabel('Damping Coefficient (gamma)')
plt.ylabel(f'Displacement x(t={t_fixed})')
plt.grid(True)
plt.legend()
plt.show()
