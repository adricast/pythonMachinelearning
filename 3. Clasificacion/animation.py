import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
line, = ax.plot(x, np.sin(x))

def update(frame):
    line.set_ydata(np.sin(x + frame / 10))  # Actualiza los datos de la línea
    return line,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()
