import numpy as np
import matplotlib.pyplot as plt

n = 10
S = np.array([2/(2*i+3) for i in range(n+1)])
A = np.array([[1/(i+j+1) for j in range(n+1)] for i in range(n+1)])

X = np.linalg.inv(A) @ S

psi_x = np.zeros(1000)
sqrt_x = np.zeros(1000)

Y = np.linspace(0,1,1000)
for j in range(1000):
    x = Y[j]
    psi_x[j] = np.sum(X[i]*x**i for i in range(n+1))
    sqrt_x[j] = np.sqrt(x)

plt.plot(Y,sqrt_x,c='r',label='sqrt_x')
plt.plot(Y,psi_x,c='b',label='psi_x')
plt.title("Comparison of two function with varying {n}")
plt.legend()
plt.show()
