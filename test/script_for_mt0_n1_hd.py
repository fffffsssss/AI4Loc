import csv
import numpy as np

# # Load the Snow CSV file
# with open('../datasets/Challenge-MT0-N1-HD/challenge-MT0-N1-HD.csv', 'r') as input_file:
#     reader = csv.reader(input_file)
#     data = list(reader)
#
# data = np.array(data)
# x = data[:, 0].astype(float)
# y = data[:, 1].astype(float)
# z = data[:, 2].astype(float)
#
# # Modify the x,y,z coordinates
# x_new_list = []
# x_new_list.append(x)
# y_new_list = []
# y_new_list.append(y)
# z_new_list = []
# z_new_list.append(z)
#
# N = 10
# for i in range(N):
#     # translation
#     x_shift = np.linspace(-1000, 2000, N)[i]
#     y_shift = np.linspace(-1000, 2000, N)[i]
#     x_new = x + x_shift
#     y_new = y + y_shift
#     z_shift = np.random.randint(-300, 300)
#     z_new = z + z_shift
#     # rotation
#     theta = np.linspace(-np.pi, np.pi, N)[i]
#     x_new = x_new * np.cos(theta) - y_new * np.sin(theta)
#     y_new = x_new * np.sin(theta) + y_new * np.cos(theta)
#     # shear transformation
#     shear = np.linspace(-0.5, 0.5, N)[i]
#     flag = np.random.randint(0, 2)
#     if flag == 0:
#         x_new = x_new + shear * y_new
#     elif flag == 1:
#         y_new = y_new + shear * x_new
#     # # mirror transformation
#     # flag = np.random.randint(0, 2)
#     # if flag == 0:
#     #     x_new = -x_new
#     # elif flag == 1:
#     #     y_new = -y_new
#
#     x_new_list.append(x_new)
#     y_new_list.append(y_new)
#     z_new_list.append(z_new)
#
# x_new_array = np.concatenate(x_new_list[:], axis=0)
# y_new_array = np.concatenate(y_new_list[:], axis=0)
# z_new_array = np.concatenate(z_new_list[:], axis=0)
# coords_new = np.array([x_new_array, y_new_array, z_new_array]).transpose()
#
# # keep the coordinates within the xy range (0, 6400)
# idx_x = np.where((coords_new[:, 0] < 0) | (coords_new[:, 0] > 6400))
# idx_y = np.where((coords_new[:, 1] < 0) | (coords_new[:, 1] > 6400))
# idx_z = np.where((coords_new[:, 2] < -700) | (coords_new[:, 2] > 700))
# idx = np.concatenate((idx_x[0], idx_y[0], idx_z[0]))
# coords_new = np.delete(coords_new, idx, axis=0)
#
# # Write the new CSV file
# with open('../datasets/Challenge-MT0-N1-HD/output.csv', 'w', newline='') as output_file:
#     writer = csv.writer(output_file)
#     writer.writerows(coords_new)




#  scale the coordinates
# Load the Snow CSV file
with open('../datasets/Challenge-MT0-N1-HD/tubulin-700.csv', 'r') as input_file:
    reader = csv.reader(input_file)
    data = list(reader)

data = np.array(data)
x = data[:, 0].astype(float)
y = data[:, 1].astype(float)
z = data[:, 2].astype(float)

x = x * 4
y = y * 4
z = z * 4
coords_new = np.array([x, y, z]).transpose()
# Write the new CSV file
with open('../datasets/Challenge-MT0-N1-HD/tubulin-2800.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(coords_new)