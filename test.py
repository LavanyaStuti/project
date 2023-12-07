import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import scipy

def createGraph(P):
    try:
        dlnay = Delaunay(P)
    except scipy.spatial.qhull.QhullError:
        raise Exception("Not enough points to construct initial simplex. Please provide a dataset with more points.")

    tri = dlnay.simplices
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)

    E = []

    for i in range(edges.shape[0]):
        dist = np.sqrt(np.sum((P[edges[i, 0], :] - P[edges[i, 1], :]) ** 2))
        E.append((edges[i, 0], edges[i, 1], dist))
    return E


# Load distance matrix from CSV
distance_matrix_file = "belt_501.csv"
distance_matrix_df = pd.read_csv(distance_matrix_file, header=None)
distance_matrix = distance_matrix_df.values

# Create the graph
G = createGraph(distance_matrix)

# Extract points from Delaunay triangulation
points = np.unique(np.array(G)[:, :2].reshape(-1))

# Visualize the graph
pos = {i: (points[i, 0], points[i, 1]) for i in range(len(points))}
edges = [(e[0], e[1]) for e in G]

# Create a graph using networkx
G_nx = nx.Graph()
G_nx.add_nodes_from(range(len(points)))
G_nx.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(G_nx, pos, with_labels=True, font_weight='bold', node_size=50, font_size=8)
plt.title('Delaunay Triangulation Graph')
plt.show()
