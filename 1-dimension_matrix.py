import gudhi, os
from gudhi.representations.metrics import pairwise_persistence_diagram_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Define labels for each row and column
labels = ['belt', 'gürtel', 'bread', 'brot', 'chair', 'stuhl', 'cloud', 'wolke', 'cup', 'tasse']

# Use pairwise_persistence_diagram_distances with the 'wasserstein' metric
distances_matrix_1d = pairwise_persistence_diagram_distances(
    X=[diag1_1d, diag2_1d, diag3_1d, diag4_1d, diag5_1d, diag6_1d, diag7_1d, diag8_1d, diag9_1d, diag10_1d],
    Y=None,
    metric='wasserstein'
)

# Set a manual range for color coding
vmin = 0  # Minimum value
vmax = 4  # Maximum value, adjust as needed

# Set diagonal values to zero
np.fill_diagonal(distances_matrix_1d, np.nan)

 # Print the matrix with words and Wasserstein distances
print("Wasserstein Distances Matrix:")
print('\t'.join(labels))
for i in range(distances_matrix_1d.shape[0]):
    print(labels[i], end='\t')
    for j in range(distances_matrix_1d.shape[1]):
        if np.isnan(distances_matrix_1d[i, j]):
            print('NaN', end='\t')
        else:
            print(f'{distances_matrix_1d[i, j]:.4f}', end='\t')
    print()

# Plot the distances matrix as an image
plt.figure(figsize=(8,8))
plt.imshow(distances_matrix_1d, cmap='viridis', origin='upper', interpolation='nearest')
plt.colorbar( fraction=0.045)
plt.title('Pairwise 1-Dimensional Wasserstein Distances Matrix')

# Set ticks and labels for both x and y axes
plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
plt.yticks(np.arange(len(labels)), labels)

# Adding annotations
for i in range(distances_matrix_1d.shape[0]):
    for j in range(distances_matrix_1d.shape[1]):
        plt.text(j, i, f'{distances_matrix_1d[i, j]:.2f}', ha='center', va='center', color='white')

# plt.tight_layout()
# Show the plot
plt.show()