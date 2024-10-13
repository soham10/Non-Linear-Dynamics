import numpy as np
import matplotlib.pyplot as plt

# Function to generate the quadratic map
def logistic_map(a, x):
    return a*x*(1-x)

# Parameters for the bifurcation diagram
a_min = 0.0   # Minimum value of 'a'
a_max = 4.0   # Maximum value of 'a'
a_steps = 10000   # Number of steps for 'a'
iterations = 1000   # Number of iterations to converge
last_iterations = 100   # Number of iterations to plot after convergence

# Generate the bifurcation diagram
a_values = np.linspace(a_min, a_max, a_steps)
x_initial = 0.5  # Initial value of x

# Store the bifurcation diagram data
bifurcation_data = []

for a in a_values:
    x = x_initial
    # Iterate to reach a stable state
    for i in range(iterations):
        x = logistic_map(a, x)
    # After reaching a stable state, store the last few iterations
    for i in range(last_iterations):
        x = logistic_map(a, x)
        bifurcation_data.append((a, x))

# Convert bifurcation data to a numpy array for plotting
bifurcation_data = np.array(bifurcation_data)

# Plot the bifurcation diagram
plt.figure(figsize=(10, 7))
plt.plot(bifurcation_data[:, 0], bifurcation_data[:, 1], ',k', alpha=0.5)
plt.title("Bifurcation Diagram of Logistic Map")
plt.xlabel("Parameter 'a'")
plt.ylabel("x")
plt.show()