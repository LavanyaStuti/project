import gudhi
from gudhi.representations.metrics import pairwise_persistence_diagram_distances
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the distance matrices from CSV
distance_matrix_file1 = 'belt_501.csv'
distance_matrix_file2 = 'bread_501.csv'
distance_matrix_file3 = 'chair_501.csv'
distance_matrix_file4 = 'cloud_501.csv'
distance_matrix_file5 = 'cup_501.csv'
distance_matrix_file6 = 'gürtel_501.csv'
distance_matrix_file7 = 'brot_501.csv'
distance_matrix_file8 = 'stuhl_501.csv'
distance_matrix_file9 = 'wolke_501.csv'
distance_matrix_file10 = 'tasse_501.csv'

distance_matrix1 = np.loadtxt(distance_matrix_file1, delimiter=',')
distance_matrix2 = np.loadtxt(distance_matrix_file2, delimiter=',')
distance_matrix3 = np.loadtxt(distance_matrix_file3, delimiter=',')
distance_matrix4 = np.loadtxt(distance_matrix_file4, delimiter=',')
distance_matrix5 = np.loadtxt(distance_matrix_file5, delimiter=',')
distance_matrix6 = np.loadtxt(distance_matrix_file6, delimiter=',')
distance_matrix7 = np.loadtxt(distance_matrix_file7, delimiter=',')
distance_matrix8 = np.loadtxt(distance_matrix_file8, delimiter=',')
distance_matrix9 = np.loadtxt(distance_matrix_file9, delimiter=',')
distance_matrix10 = np.loadtxt(distance_matrix_file10, delimiter=',')

# Create Rips complexes
rips_complex1 = gudhi.RipsComplex(distance_matrix=distance_matrix1, max_edge_length=0.7)
rips_complex2 = gudhi.RipsComplex(distance_matrix=distance_matrix2, max_edge_length=0.7)
rips_complex3 = gudhi.RipsComplex(distance_matrix=distance_matrix3, max_edge_length=0.7)
rips_complex4 = gudhi.RipsComplex(distance_matrix=distance_matrix4, max_edge_length=0.7)
rips_complex5 = gudhi.RipsComplex(distance_matrix=distance_matrix5, max_edge_length=0.7)
rips_complex6 = gudhi.RipsComplex(distance_matrix=distance_matrix6, max_edge_length=0.7)
rips_complex7 = gudhi.RipsComplex(distance_matrix=distance_matrix7, max_edge_length=0.7)
rips_complex8 = gudhi.RipsComplex(distance_matrix=distance_matrix8, max_edge_length=0.7)
rips_complex9 = gudhi.RipsComplex(distance_matrix=distance_matrix9, max_edge_length=0.7)
rips_complex10 = gudhi.RipsComplex(distance_matrix=distance_matrix10, max_edge_length=0.7)

# Create simplex trees
simplex_tree1 = rips_complex1.create_simplex_tree(max_dimension=2)
simplex_tree2 = rips_complex2.create_simplex_tree(max_dimension=2)
simplex_tree3 = rips_complex3.create_simplex_tree(max_dimension=2)
simplex_tree4 = rips_complex4.create_simplex_tree(max_dimension=2)
simplex_tree5 = rips_complex5.create_simplex_tree(max_dimension=2)
simplex_tree6 = rips_complex6.create_simplex_tree(max_dimension=2)
simplex_tree7 = rips_complex7.create_simplex_tree(max_dimension=2)
simplex_tree8 = rips_complex8.create_simplex_tree(max_dimension=2)
simplex_tree9 = rips_complex9.create_simplex_tree(max_dimension=2)
simplex_tree10 = rips_complex10.create_simplex_tree(max_dimension=2)

# Compute persistence diagrams for dimension 1
simplex_tree1.compute_persistence()
simplex_tree2.compute_persistence()
simplex_tree3.compute_persistence()
simplex_tree4.compute_persistence()
simplex_tree5.compute_persistence()
simplex_tree6.compute_persistence()
simplex_tree7.compute_persistence()
simplex_tree8.compute_persistence()
simplex_tree9.compute_persistence()
simplex_tree10.compute_persistence()

# Extract persistence diagrams for 1-dimensional persistence
diag1_1d = simplex_tree1.persistence_intervals_in_dimension(1)
diag2_1d = simplex_tree2.persistence_intervals_in_dimension(1)
diag3_1d = simplex_tree3.persistence_intervals_in_dimension(1)
diag4_1d = simplex_tree4.persistence_intervals_in_dimension(1)
diag5_1d = simplex_tree5.persistence_intervals_in_dimension(1)
diag6_1d = simplex_tree6.persistence_intervals_in_dimension(1)
diag7_1d = simplex_tree7.persistence_intervals_in_dimension(1)
diag8_1d = simplex_tree8.persistence_intervals_in_dimension(1)
diag9_1d = simplex_tree9.persistence_intervals_in_dimension(1)
diag10_1d = simplex_tree10.persistence_intervals_in_dimension(1)

# # Ensure that the diagrams are in the correct format
# diag1_1d = [(float(interval[0]), float(interval[1])) for interval in diag1_1d]
# diag2_1d = [(float(interval[0]), float(interval[1])) for interval in diag2_1d]
# diag3_1d = [(float(interval[0]), float(interval[1])) for interval in diag3_1d]
# diag4_1d = [(float(interval[0]), float(interval[1])) for interval in diag4_1d]
# diag5_1d = [(float(interval[0]), float(interval[1])) for interval in diag5_1d]
# diag6_1d = [(float(interval[0]), float(interval[1])) for interval in diag6_1d]
# diag7_1d = [(float(interval[0]), float(interval[1])) for interval in diag7_1d]
# diag8_1d = [(float(interval[0]), float(interval[1])) for interval in diag8_1d]
# diag9_1d = [(float(interval[0]), float(interval[1])) for interval in diag9_1d]
# diag10_1d = [(float(interval[0]), float(interval[1])) for interval in diag10_1d]

# Define labels for each row and column
labels = ['belt', 'bread', 'chair', 'cloud', 'cup', 'tasse', 'brot', 'wolke', 'gürtel', 'stuhl']

# Use pairwise_persistence_diagram_distances with the 'wasserstein' metric for 1-dimensional
distances_matrix_1d = pairwise_persistence_diagram_distances(
    X=[diag1_1d, diag2_1d, diag3_1d, diag4_1d, diag5_1d, diag6_1d, diag7_1d, diag8_1d, diag9_1d, diag10_1d],
    Y=None,
    metric='wasserstein'
)

# Set diagonal values to zero
np.fill_diagonal(distances_matrix_1d, 0)

# Replace "inf" values with a finite value (e.g., 500)
finite_value = 500
distances_matrix_1d[np.isinf(distances_matrix_1d)] = finite_value

# Set a manual range for color coding
vmin = 0  # Minimum value
vmax = 4  # Maximum value, adjust as needed

# Create Pandas DataFrame
df_distances = pd.DataFrame(distances_matrix_1d, index=labels, columns=labels)

# Display the DataFrame
print("\n1-Dimensional Wasserstein Distances Matrix:")
print(df_distances)

# Plot the distances matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_distances, cmap='viridis', annot=True, fmt=".4f", vmin=vmin, vmax=vmax)

plt.title('Pairwise 1-Dimensional Wasserstein Distances Matrix')
plt.show()