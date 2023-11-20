import matplotlib.pyplot as plt
import gudhi
import numpy as np

# Read the distance matrix from CSV
distance_matrix_file = 'auge_101.csv'
distance_matrix = np.loadtxt(distance_matrix_file, delimiter=',')

# Create a Rips complex based on the distance matrix
rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=0.7)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
#diag = simplex_tree.persistence(min_persistence=0.4)
diag = simplex_tree.persistence()

# Plot the persistence barcode
gudhi.plot_persistence_barcode(diag)
plt.show()
