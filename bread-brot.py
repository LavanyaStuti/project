import gudhi
from gudhi.representations.metrics import pairwise_persistence_diagram_distances
import numpy as np
import matplotlib.pyplot as plt

# Read the distance matrices from CSV
distance_matrix_file1 = 'belt_501.csv'
distance_matrix_file2 = 'bread_501.csv'
distance_matrix_file3 = 'chair_501.csv'
distance_matrix_file4 = 'cloud_501.csv'
distance_matrix_file5 = 'cup_501.csv'
distance_matrix_file6 = 'tasse_501.csv'
distance_matrix_file7 = 'brot_501.csv'
distance_matrix_file8 = 'wolke_501.csv'
distance_matrix_file9 = 'gürtel_501.csv'
distance_matrix_file10 = 'stuhl_501.csv'

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
simplex_tree1 = rips_complex1.create_simplex_tree(max_dimension=1)
simplex_tree2 = rips_complex2.create_simplex_tree(max_dimension=1)
simplex_tree3 = rips_complex3.create_simplex_tree(max_dimension=1)
simplex_tree4 = rips_complex4.create_simplex_tree(max_dimension=1)
simplex_tree5 = rips_complex5.create_simplex_tree(max_dimension=1)
simplex_tree6 = rips_complex6.create_simplex_tree(max_dimension=1)
simplex_tree7 = rips_complex7.create_simplex_tree(max_dimension=1)
simplex_tree8 = rips_complex8.create_simplex_tree(max_dimension=1)
simplex_tree9 = rips_complex9.create_simplex_tree(max_dimension=1)
simplex_tree10 = rips_complex10.create_simplex_tree(max_dimension=1)

# Compute persistence diagrams

diag1 = simplex_tree1.persistence()
diag2 = simplex_tree2.persistence()
diag3 = simplex_tree3.persistence()
diag4 = simplex_tree4.persistence()
diag5 = simplex_tree5.persistence()
diag6 = simplex_tree6.persistence()
diag7 = simplex_tree7.persistence()
diag8 = simplex_tree8.persistence()
diag9 = simplex_tree9.persistence()
diag10 = simplex_tree10.persistence()

nt1 = [point[1] for point in diag1]
nt2 = [point[1] for point in diag2]
nt3 = [point[1] for point in diag3]
nt4 = [point[1] for point in diag4]
nt5 = [point[1] for point in diag5]
nt6 = [point[1] for point in diag6]
nt7 = [point[1] for point in diag7]
nt8 = [point[1] for point in diag8]
nt9 = [point[1] for point in diag9]
nt10 = [point[1] for point in diag10]

dnp1 = np.array(nt1)
dnp2 = np.array(nt2)
dnp3 = np.array(nt3)
dnp4 = np.array(nt4)
dnp5 = np.array(nt5)
dnp6 = np.array(nt6)
dnp7 = np.array(nt7)
dnp8 = np.array(nt8)
dnp9 = np.array(nt9)
dnp10 = np.array(nt10)


# Use pairwise_persistence_diagram_distances with the 'wasserstein' metric
distances_matrix = pairwise_persistence_diagram_distances(
     X=[dnp1, dnp2, dnp3, dnp4, dnp5, dnp6, dnp7, dnp8, dnp9, dnp10], Y=None, metric='wasserstein'
 )


# Print the computed distances matrix
print("Wasserstein Distances Matrix:")
print(distances_matrix)

# Plot the distances matrix as an image
plt.imshow(distances_matrix, cmap='viridis', origin='upper', interpolation='nearest')
plt.colorbar()
plt.title('Pairwise Wasserstein Distances Matrix')
plt.show()
