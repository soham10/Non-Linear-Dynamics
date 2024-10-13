import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x, A):
    return A * x * (1 - x)

x0 = 0.1
x1 = 0.1 + 1e-6
A = 3.66
n_iterations = 100

def trajectory(n_iterations,A,x0):
    X = np.zeros(n_iterations + 1)
    X[0] = x0
    step = []

    for i in range(n_iterations):
        X[i + 1] = logistic_map(X[i], A)
        step.append(i)
    return X

#print(X)
plt.plot(trajectory(n_iterations,A,x0))
plt.plot(trajectory(n_iterations,A,x1),c='red')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Logistic Map')
plt.show()