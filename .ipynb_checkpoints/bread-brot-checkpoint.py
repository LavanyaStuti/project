import gudhi
import numpy as np
from gudhi.representations.metrics import pairwise_persistence_diagram_distances


distance_matrix_file = 'bread_501.csv'
distance_matrix_file2 = 'brot_501.csv'
distance_matrix = np.loadtxt(distance_matrix_file, delimiter=',')
distance_matrix2 = np.loadtxt(distance_matrix_file2, delimiter=',')


rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=0.5)
rips_complex2 = gudhi.RipsComplex(distance_matrix=distance_matrix2, max_edge_length=0.5)


simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
simplex_tree2 = rips_complex2.create_simplex_tree(max_dimension=1)


diag = simplex_tree.persistence()
diag2 = simplex_tree2.persistence()

# d1 = np.array([[point[0], point[1]] for point in diag if len(point) == 2])
# d2 = np.array([[point[0], point[1]] for point in diag2 if len(point) == 2])

a = np.array([0,1], [1,2])

distances_matrix = pairwise_persistence_diagram_distances(
    X=[a], Y=[a], metric='wasserstein'
)


print("Wasserstein Distances Matrix:")
print(distances_matrix) 