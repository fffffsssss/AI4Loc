import numpy as np
import matplotlib.pyplot as plt

# Define a simple curve, y = sin(x)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Plot the original curve
plt.plot(x, y, label='Original')

# Randomly reshape the curve
random_factor = 0.2  # Adjust this to change the amount of reshaping
y_reshaped = y + random_factor * np.random.randn(len(y))

# Plot the reshaped curve
plt.plot(x, y_reshaped, label='Reshaped')

plt.legend()
plt.show()


