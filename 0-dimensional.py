import numpy as np
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple
import numpy.typing as npt
import pandas as pd
from networkx import Graph, draw
import matplotlib.pyplot as plt

def source(C: Dict[int, int], v: int):
    while v != C[v]:
        v = C[v]
    return v

def persistenceDiagram(D: npt.ArrayLike, k: int, return_filtration: bool = False) -> np.array | Tuple[np.array, np.array, List[Tuple[int, int, float]]]:
    # TODO: Add checks and error checking
    mst = minimum_spanning_tree(D)
    
    rows, cols = mst.nonzero()
    edges = [(i, j, mst[i, j]) for i, j in zip(rows, cols)]
    edges.sort(key=lambda x: x[2])
    
    verts = list(range(D.shape[0]))
    conn = dict(zip(verts, verts))
    weights = dict.fromkeys(verts, 1)
    F = np.zeros(len(verts))
    PD = []

    if return_filtration:
        non_active = {v: set([v, ]) for v in verts}

    for x, y, f in edges:
        x_source = source(conn, x)
        y_source = source(conn, y)
        
        s = x_source if x_source < y_source else y_source
        t = y_source if x_source < y_source else x_source
        total_weight = weights[s] + weights[t]

        if (weights[s] < k) and (weights[t] < k):
            weights[s] = total_weight
            weights[t] = 0
            conn[t] = s
            
            if return_filtration:
                non_active[s].update(non_active[t])

            if weights[s] >= k:
                if return_filtration:
                    for v in non_active[s]:
                        F[v] = f
                else:
                    F[s] = f

        elif (weights[s] >= k) and (weights[t] < k):
            weights[s] += weights[t]
            weights[t] = 0
            conn[t] = s

            if return_filtration:
                for v in non_active[t]:
                    F[v] = f

        elif (weights[s] < k) and (weights[t] >= k):
            weights[t] += weights[s]
            weights[s] = 0
            conn[s] = t

            if return_filtration:
                for v in non_active[s]:
                    F[v] = f

        elif (weights[s] >= k) and (weights[t] >= k):
            w = s if F[s] <= F[t] else t
            v = t if F[s] <= F[t] else s
            weights[w] += weights[v]
            weights[v] = 0
            conn[v] = w 
            PD.append((F[v], f))
        else:
            raise Exception("No case matches - bug")
    
    inf_bars = set([source(conn, i) for i in verts])
    PD += [(F[j], np.inf) for j in inf_bars]

    if return_filtration:
        E = [(e[0], e[1], max([e[2], F[e[0]], F[e[1]]])) for e in edges]
        E.sort(key=lambda x: x[2])
        return np.array(PD), F, E
    else:    
        return np.array(PD)
    
# # Load the distance matrix from CSV file
# distance_matrix_df = pd.read_csv('belt_501.csv', header=None)
# distance_matrix = distance_matrix_df.values

# # Set the filtration parameter k
# k_value = 2

# # Compute persistence diagram
# persistence_diagram = persistenceDiagram(distance_matrix, k=k_value)

# # Print the persistence diagram
# print("Persistence Diagram:")
# print(persistence_diagram)

def comp(a: Tuple[int, float], b: Tuple[int, float]) -> bool:
    if a[1] != b[1]:
        return a[1] < b[1]
    else:
        return a[0] < b[0]

def getClusters(F: np.array, E: Tuple[int, int, float], alpha: float, multiplicative: bool = True) -> Tuple[np.array, Dict[int, int]]:
    verts = list(range(F.shape[0]))
    conn = dict(zip(verts, verts))
    E.sort(key=lambda x: x[2])

    for x, y, f in E:
        x_source = source(conn, x)
        y_source = source(conn, y)
        s = x_source if comp((x_source, F[x_source]), (y_source, F[y_source])) else y_source
        t = y_source if comp((x_source, F[x_source]), (y_source, F[y_source])) else x_source

        if multiplicative:
            if f / F[t] < alpha:
                conn[t] = s
        else:
            if f - F[t] < alpha:
                conn[t] = s

    clstrs = np.zeros(len(verts))
    clstr_heads = set()

    for v in conn.keys():
        clstrs[v] = source(conn, v)
        clstr_heads.add(clstrs[v])
    
    clstr_id = dict(zip(clstr_heads, range(len(clstr_heads))))

    return clstrs, clstr_id

def getThreshold(PD: np.array, m: int, multiplicative: bool = True) -> float:
    if multiplicative and (np.count_nonzero(PD[:, 0]) < PD.shape[0]):
        raise Exception("Cannot use multiplicative if vertex can take value 0")
    
    pers_values = PD[:, 1] / PD[:, 0] if multiplicative else PD[:, 1] - PD[:, 0]
    pers_values = np.sort(pers_values)[::-1]

    return (pers_values[m] + pers_values[m - 1]) / 2

def computeMST(E: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    E.sort(key=lambda x: x[2])
    verts = list(set([e[0] for e in E] + [e[1] for e in E]))
    conn = dict(zip(verts, verts))
    MST = []

    for x, y, f in E:
        x_source = source(conn, x)
        y_source = source(conn, y)

        if x_source != y_source:
            s = x_source if x_source < y_source else y_source
            t = y_source if x_source < y_source else x_source 
            conn[t] = s
            MST.append((x, y, f))
                        
    return MST

def persistenceDiagramGraph(edges: List[Tuple[int, int, float]], k: int, return_filtration: bool = False) -> np.array | Tuple[np.array, np.array, List[Tuple[int, int, float]]]:
    edges.sort(key=lambda x: x[2])
    verts = list(set([e[0] for e in edges] + [e[1] for e in edges]))
    conn = dict(zip(verts, verts))
    weights = dict.fromkeys(verts, 1)
    F = dict.fromkeys(verts, 0)
    PD = []

    if return_filtration:
        non_active = {v: set([v, ]) for v in verts}

    for x, y, f in edges:
        x_source = source(conn, x)
        y_source = source(conn, y)
        s = x_source if x_source < y_source else y_source
        t = y_source if x_source < y_source else x_source

        total_weight = weights[s] + weights[t]

        if (weights[s] < k) and (weights[t] < k):
            weights[s] = total_weight
            weights[t] = 0
            conn[t] = s
            
            if return_filtration:
                non_active[s].update(non_active[t])

            if weights[s] >= k:
                if return_filtration:
                    for v in non_active[s]:
                        F[v] = f
                else:
                    F[s] = f

        elif (weights[s] >= k) and (weights[t] < k):
            weights[s] += weights[t]
            weights[t] = 0
            conn[t] = s

            if return_filtration:
                for v in non_active[t]:
                    F[v] = f

        elif (weights[s] < k) and (weights[t] >= k):
            weights[t] += weights[s]
            weights[s] = 0
            conn[s] = t

            if return_filtration:
                for v in non_active[s]:
                    F[v] = f

        elif (weights[s] >= k) and (weights[t] >= k):
            w = s if F[s] <= F[t] else t
            v = t if F[s] <= F[t] else s
            weights[w] += weights[v]
            weights[v] = 0
            conn[v] = w 
            PD.append((F[v], f))
        else:
            raise Exception("No case matches - bug")
    
    inf_bars = set([source(conn, i) for i in verts])
    PD += [(F[j], np.inf) for j in inf_bars]

    if return_filtration:
        E = [(e[0], e[1], max([e[2], F[e[0]], F[e[1]]])) for e in edges]
        E.sort(key=lambda x: x[2])
        return np.array(PD), F, E
    else:    
        return np.array(PD)

def getClustersGraph(F: Dict[int, float], E: Tuple[int, int, float], alpha: float, multiplicative: bool = True) -> Tuple[np.array, Dict[int, int]]:
    F_tmp = np.array([F[i] for i in F.keys()])
    if multiplicative and (np.count_nonzero(F_tmp) < F_tmp.shape[0]):
        raise Exception("Cannot use multiplicative if vertex can take value 0")
 
    verts = list(F.keys())
    conn = dict(zip(verts, verts))
    E.sort(key=lambda x: x[2])

    for x, y, f in E:
        x_source = source(conn, x)
        y_source = source(conn, y)
        s = x_source if comp((x_source, F[x_source]), (y_source, F[y_source])) else y_source
        t = y_source if comp((x_source, F[x_source]), (y_source, F[y_source])) else x_source

        if multiplicative:
            if f / F[t] < alpha:
                conn[t] = s
        else:
            if f - F[t] < alpha:
                conn[t] = s

    clstrs = np.zeros(len(verts))
    clstr_heads = set()

    for v in conn.keys():
        clstrs[v] = source(conn, v)
        clstr_heads.add(clstrs[v])
    
    clstr_id = dict(zip(clstr_heads, range(len(clstr_heads))))

    return clstrs, clstr_id

def plotClusterGraph(distance_matrix: np.array, clusters: np.array, cluster_ids: Dict[int, int]):
    G = Graph()

    # Add nodes to the graph
    for node in range(distance_matrix.shape[0]):
        G.add_node(node)

    # Add edges to the graph
    for i, row in enumerate(distance_matrix):
        for j, weight in enumerate(row):
            G.add_edge(i, j, weight=weight)

    # Plot the graph with nodes colored by clusters
    node_colors = [cluster_ids[cluster] for cluster in clusters]
    draw(G, with_labels=True, node_color=node_colors, cmap=plt.cm.Blues)
    plt.show()

# Load the distance matrix from CSV file
distance_matrix_df = pd.read_csv('belt_501.csv', header=None)
distance_matrix = distance_matrix_df.values

# Set the filtration parameter k
k_value = 2

# Compute persistence diagram
persistence_diagram, vertex_filtration, edges_filtration = persistenceDiagram(distance_matrix, k=k_value, return_filtration=True)

# Get clusters from persistence diagram
alpha_value = 0.5  # Set the threshold parameter alpha
clusters, cluster_ids = getClusters(vertex_filtration, edges_filtration, alpha=alpha_value)

# Plot the cluster graph
plotClusterGraph(distance_matrix, clusters, cluster_ids)

# Print the clusters
print("Clusters:")
print(clusters)
