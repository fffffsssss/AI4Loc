
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data

# Load the image
image = data.coins()

# Find the coordinates of local maxima
coordinates = peak_local_max(image, min_distance=20)

# Display the results
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.plot(coordinates[:, 1], coordinates[:, 0], 'rx')
# ax.axis('off')
plt.show()