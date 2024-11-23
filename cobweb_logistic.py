import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots()
x_initial = 0.1
x = np.linspace(0, 1, 400)
identity_line, = ax.plot(x, x, 'r-', label='Identity Line')
quadratic_map_line, = ax.plot([], [], 'b-', label='Quadratic Map')
cobweb_lines, = ax.plot([], [], 'g-', alpha=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('x_n')
ax.set_ylabel('x_(n+1)')
plt.legend()

def init():
    quadratic_map_line.set_data([], [])
    cobweb_lines.set_data([], [])
    return quadratic_map_line, cobweb_lines

def update(A):
    y = A * x * (1 - x)
    quadratic_map_line.set_data(x, y)
    
    x_vals = [x_initial]
    y_vals = [0]
    x_n = x_initial
    for _ in range(100):
        x_np1 = A * x_n * (1 - x_n)
        x_vals.extend([x_n, x_n])
        y_vals.extend([x_n, x_np1])
        x_vals.append(x_np1)
        y_vals.append(x_np1)
        x_n = x_np1
        if x_np1 > 1 or x_np1 < 0:  # Prevent going outside bounds
            break
    
    cobweb_lines.set_data(x_vals, y_vals)
    ax.set_title(f'A = {A:.2f}')
    return quadratic_map_line, cobweb_lines

ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 4, 200), init_func=init, blit=True)

plt.title('Cobweb Plot of Quadratic Map x_(n+1) = A*x_n*(1-x_n)')
plt.show()


