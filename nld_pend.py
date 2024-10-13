import math
import matplotlib.pyplot as plt

# Parameters for the pendulum
g = 9.81  # acceleration due to gravity (m/s^2)
l = 1.0   # length of the pendulum (m)
b = 0.05  # damping coefficient (adjust for non-linearity)

# Time step for the Runge-Kutta method
dt = 0.01

# Define the non-linear pendulum equations
def pendulum_derivatives(state, t):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = - (g / l) * (theta - (theta**3)/3)
    return dtheta_dt, domega_dt

# Runge-Kutta 4th order method
def runge_kutta_4(state, t, dt):
    k1 = [dt * d for d in pendulum_derivatives(state, t)]
    k2 = [dt * d for d in pendulum_derivatives([state[0] + k1[0] / 2, state[1] + k1[1] / 2], t + dt / 2)]
    k3 = [dt * d for d in pendulum_derivatives([state[0] + k2[0] / 2, state[1] + k2[1] / 2], t + dt / 2)]
    k4 = [dt * d for d in pendulum_derivatives([state[0] + k3[0], state[1] + k3[1]], t + dt)]
    
    theta_new = state[0] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
    omega_new = state[1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
    
    return theta_new, omega_new

# Initial conditions
theta = math.pi / 4  # initial angle (radians)
omega = 0.0          # initial angular velocity
state = (theta, omega)
t = 0.0

# Lists to store the results
times = []
positions = []

# Main simulation loop
while t < 10000:  # simulate for 10 seconds
    # Calculate new state using Runge-Kutta method
    state = runge_kutta_4(state, t, dt)
    theta, omega = state
    
    # Store the time and position
    times.append(t)
    x = l * math.sin(theta)
    y = -l * math.cos(theta)
    positions.append((x, y))
    
    # Update the time
    t += dt

# Extract x and y coordinates for plotting
x_coords, y_coords = zip(*positions)

# Plot the trajectory
plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, label='Pendulum Path')
plt.title('Pendulum Trajectory in Phase Space')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.grid(True)
plt.show()
