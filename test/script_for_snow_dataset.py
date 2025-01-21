import csv
import numpy as np

# Load the Snow CSV file
with open('../datasets/Challenge-Snow/fluos.csv', 'r') as input_file:
    reader = csv.reader(input_file)
    data = list(reader)

data = np.array(data)
x_snow = data[1:, 2].astype(float)
y_snow = data[1:, 3].astype(float)
z_snow = data[1:, 4].astype(float)

# Modify the x,y,z coordinates
x_snow_new = x_snow/10
y_snow_new = y_snow/10
x_snow_new += 1000
y_snow_new += 1000
z_snow_new = (z_snow-200)/2*6  # (-600, -300, 0 ,300, 600)

coords_new = np.stack((x_snow_new, y_snow_new, z_snow_new), axis=1)

# Write the new CSV file
with open('../datasets/Challenge-Snow/output.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(coords_new)
