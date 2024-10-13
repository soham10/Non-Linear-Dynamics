import pygame
import numpy as np
import matplotlib.pyplot as plt

theta_init = np.pi / 2  # initial angle (radians)
omega_init = 0.0  # initial angular velocity (rad/s)

# Simulation parameters
dt = 0.01  # time step (s)
t_max = 10  # total simulation time (s)

# Define gamma and omega_0 values (can be adjusted)
gamma_underdamped = 0.5  # damping coefficient for underdamped
gamma_critical = 3.0  # damping coefficient for critically damped (usually gamma = omega_0)
gamma_overdamped = 6  # damping coefficient for overdamped

omega_0 = 3.0  # natural frequency of the system (omega_0^2 = g/L in the case of a pendulum)

# Damping conditions dictionary
damping_conditions = {
    "Underdamped": gamma_underdamped,
    "Critically damped": gamma_critical,
    "Overdamped": gamma_overdamped
}

# Initialize pygame
pygame.init()

def rk4_step(theta, omega, gamma, omega_0, dt):
    def derivatives(theta, omega):
        dtheta_dt = omega
        domega_dt = -omega_0**2 * np.sin(theta) - 2 * gamma * omega
        return dtheta_dt, domega_dt

    k1_theta, k1_omega = derivatives(theta, omega)
    k2_theta, k2_omega = derivatives(theta + 0.5 * dt * k1_theta, omega + 0.5 * dt * k1_omega)
    k3_theta, k3_omega = derivatives(theta + 0.5 * dt * k2_theta, omega + 0.5 * dt * k2_omega)
    k4_theta, k4_omega = derivatives(theta + dt * k3_theta, omega + dt * k3_omega)

    theta_new = theta + (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_new = omega + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

    return theta_new, omega_new

def simulate_pendulum(gamma, omega_0, condition_name):
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"Pendulum Simulation - {condition_name}")
    clock = pygame.time.Clock()
    
    theta = theta_init
    omega = omega_init
    t = 0
    time_data = []
    theta_data = []

    while t < t_max:
        theta, omega = rk4_step(theta, omega, gamma, omega_0, dt)
        t += dt

        time_data.append(t)
        theta_data.append(theta)

        # Update Pygame display
        screen.fill((255, 255, 255))

        # Convert to screen coordinates
        x = int(400 + np.sin(theta) * 100)
        y = int(300 + np.cos(theta) * 100)
        pygame.draw.line(screen, (0, 0, 0), (400, 300), (x, y), 2)
        pygame.draw.circle(screen, (0, 0, 255), (x, y), 15)

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    pygame.display.quit()  # Close the current window
    return time_data, theta_data

# Run simulations for each damping condition and store results
all_time_data = []
all_theta_data = []
for condition, gamma in damping_conditions.items():
    time_data, theta_data = simulate_pendulum(gamma, omega_0, condition)
    all_time_data.append(time_data)
    all_theta_data.append(theta_data)

# Plot results
plt.figure(figsize=(12,10))
for i, condition in enumerate(damping_conditions.keys()):
    plt.plot(all_time_data[i], all_theta_data[i], label=condition)

plt.title('Pendulum Motion under Different Damping Conditions (RK4 Method)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.show()

pygame.quit()
