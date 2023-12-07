import gudhi
from gudhi.representations.metrics import pairwise_persistence_diagram_distances
import numpy as np
import matplotlib.pyplot as plt

def load_distance_matrix(file_path):
    return np.loadtxt(file_path, delimiter=',')

def create_rips_complex(distance_matrix, max_edge_length=0.7):
    return gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)

def create_simplex_tree(rips_complex, max_dimension=2):
    return rips_complex.create_simplex_tree(max_dimension=max_dimension)

def compute_persistence(tree):
    tree.compute_persistence()

def persistence_intervals(tree, dimension):
    return tree.persistence_intervals_in_dimension(dimension)

def calculate_pairwise_distances(diagrams, metric='wasserstein'):
    return pairwise_persistence_diagram_distances(X=diagrams, Y=None, metric=metric)


# List of file paths
file_paths = ['belt_501.csv', 'bread_501.csv', 'chair_501.csv', 'cloud_501.csv', 'cup_501.csv',
              'gürtel_501.csv', 'brot_501.csv', 'stuhl_501.csv', 'wolke_501.csv', 'tasse_501.csv']

# Load distance matrices and create Rips complexes and simplex trees
rips_complexes = [create_rips_complex(load_distance_matrix(file)) for file in file_paths]
simplex_trees = [create_simplex_tree(rips_complex) for rips_complex in rips_complexes]

# Compute persistence and extract persistence diagrams for 1-dimensional persistence
for tree in simplex_trees:
    compute_persistence(tree)

diagrams_1d = [persistence_intervals(tree, dimension=1) for tree in simplex_trees]

# Calculate pairwise distances
distances_matrix_1d = calculate_pairwise_distances(diagrams_1d)

# Define labels for each row and column
labels = ['belt', 'bread', 'chair', 'cloud', 'cup', 'tasse', 'brot', 'wolke', 'gürtel', 'stuhl']

# Set diagonal values to NaN
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