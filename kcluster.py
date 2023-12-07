############################################################
#
#  Library for k-cluster filtration
# 
############################################################

# Reqirements numpy, scipy, and typing (so it works for python before 3.9)

import numpy as np
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

# for readability
from typing import Dict, List, Tuple
import numpy.typing as npt



############################################################
# helper function
# - this should never be called directly
# - follows the 
############################################################
def source(C:Dict[int,int],v:int):
    while (v!=C[v]):
        v=C[v]
    return v


############################################################
# Compute persistence diagram 
# - input D: distance matrix 
#         k: parameter for filtration
#            return_filtration: whether to return the edges filtration function 
#            (needed for clustering but with some overhead)
# - returns: if return_filtration is False
#                PD: numpy array with 1st column birth second column death
#            if return_filtration is True
#                PD: as above
#                F: numpy array with vertex filtration values
#                E: list of tuples with each tuple being one edge 
#                          (edge source, edge target, edge weight)  
# Notes: 1. this does not return points
#           on the diagonal,
#        2. this is technically a 2-pass algorithm, since we 
#           use scipy's MST implementation
# TODO: Make wrapper functions for a cleaner pattern
############################################################
def persistenceDiagram(D:npt.ArrayLike,k:int, return_filtration:bool=False)->np.array | Tuple[np.array,np.array,List[Tuple[int,int,float]]]:
    # TODO: Add checks and errorchecking
    mst = minimum_spanning_tree(D)    # compute MST
    
    # create ordered list of edges
    rows,cols = mst.nonzero()
    edges = [(i,j,mst[i,j]) for i,j in zip(rows,cols)]
    edges.sort(key = lambda x: x[2])
    
    # Initialization 
    verts = list(range(D.shape[0]))        # make list of vertices
    conn = dict(zip(verts,verts))          # simplified union find 
                                           # connected components look up table

    weights = dict.fromkeys(verts,1)       # weight function
    F = np.zeros(len(verts))               # vertex filtration function
    PD = []                                # initialize empty persistence diagram

    if return_filtration:
        non_active = {v:set([v,]) for v in verts}   # to keep track of non-active 
                                                    # components to update filtration

    #------------------------------------------
    # Main loop - iterate over edges in order
    # - this is massively simplified since 
    #   all edges are negative 
    #------------------------------------------
    for x,y,f in edges:  # edges is (x,y) with edge weight f
       
        #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # apply lexicographical ordering (to prevent further tests)
        # NOTE: since these are representative vertices, we cannot 
        #       control the ordering a priori
        s = x_source if x_source<y_source else y_source
        t = y_source if x_source<y_source else x_source

        total_weight = weights[s]+weights[t]     # total weight of new component

        #---------------------------------------
        # Cases: 
        #---------------------------------------
        
        # neither component active
        if (weights[s]<k) and (weights[t]<k):
            # update weights to new root
            weights[s] = total_weight
            weights[t] = 0                               # for debugging purposes

            conn[t] = s                                  # update union find root
            
            if return_filtration:
                non_active[s].update(non_active[t])      # the pop empties the list and copies
                                                         # it into the root list for the component

            # if it has become active then record 
            # function of root vertex 
            if (weights[s]>=k):
                if return_filtration:                    # update entire component
                    for v in non_active[s]:
                        F[v] = f 
                else:
                     F[s] = f                            # update only root 

        # s component active, tnot
        elif (weights[s]>=k) and (weights[t]<k):
            weights[s] += weights[t]
            weights[t] = 0                               # for debugging purposes
            
            conn[t] = s                                  # update union find root
 
            if return_filtration:                        # update nonactive component
                for v in non_active[t]:
                    F[v] = f
              
        # t component active, s not (this is verbose but simple)
        elif (weights[s]<k) and (weights[t]>=k):
            weights[t] += weights[s]
            weights[s] = 0               # for debugging purposes
            
            conn[s] = t                  # update union find root

            if return_filtration:        # update nonactive component
                for v in non_active[s]:
                    F[v] = f

        # both components active
        elif (weights[s]>=k) and (weights[t]>=k):
            # F is defined here for both
            # redefine vertices wrt function value F
            w = s if F[s]<=F[t] else t
            v = t if F[s]<=F[t] else s
            
            weights[w] += weights[v]
            weights[v] = 0               # for debugging purposes
            
            conn[v] = w                  # update union find root 
            
            PD.append((F[v],f))          # add point to persistence diagram
        else:
            raise Exception("No case matches - bug")
    #------------------------------------------

    inf_bars = set([source(conn,i) for i in verts])   # find infinite components
    PD+=[(F[j],np.inf) for j in inf_bars]

    if return_filtration:
        E = [(e[0],e[1], max([e[2],F[e[0]],F[e[1]]])) for e in edges]
        E.sort(key = lambda x: x[2])
        return np.array(PD),F,E
    else:    
        return np.array(PD)


############################################################
# Custom comparison function - should never 
# be called directly
# 
############################################################
def comp(a: Tuple[int,float],b: Tuple[int,float])->bool:
    if a[1]!=b[1]:
        return a[1]<b[1]
    else:
        return a[0]<b[0]
    
############################################################
# Get clusters - assumes persistence diagram computed
#     
# - input F: numpy array with filtration values for vertices
#         E: edges of MST with weights
#         alpha: threshold parameter
#         multiplicative: whether death/birth (True) is used 
#                         or death - birth (False)
#
#         Note that if birth is 0 and multiplicative is used
#         an exception will be raised
# - returns: 
#         clstrs: numpy array with representative vertex for 
#                 each cluster
#         clstr_id: a dictionary with a numbering for the 
#                   clusters so if you wanted the clusters
#                   named 1,2,3...
#                   [clstr_id[i] for i in clstrs]
############################################################
def getClusters(F:np.array, E:Tuple[int,int,float], alpha:float, multiplicative:bool=True)->Tuple[np.array,Dict[int,int]]:
    
    # test that no vertices have value 0
    if multiplicative and (np.count_nonzero(F)<F.shape[0]):
        raise Exception("Cannot use multiplicative if vertex can take value 0")
 
    verts = list(range(F.shape[0]))          # make list of vertices
    conn = dict(zip(verts,verts))            # simplified union find 

    E.sort(key = lambda x: x[2]) # re-sort edges just in case

    # main loop
    for x,y,f in E:
         #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # apply function and in case of equality lexicographical ordering
        # (to prevent further tests)
        # NOTE: since these are representative vertices, we cannot 
        #       control the ordering a priori
        s = x_source if comp((x_source,F[x_source]),(y_source,F[y_source])) else y_source
        t = y_source if comp((x_source,F[x_source]),(y_source,F[y_source])) else x_source

        # all edges are negative so only need to check threshold
        # whether to merge
        if multiplicative:
            if f/F[t]<alpha:
                conn[t]=s
        else:
            if f-F[t]<alpha:
                conn[t]=s

    # turn dictionary into np.array
    clstrs = np.zeros(len(verts))
    clstr_heads = set()
    for v in conn.keys():
        clstrs[v] = source(conn,v)
        clstr_heads.add(clstrs[v])
    
    clstr_id = dict(zip(clstr_heads,range(len(clstr_heads))))

    return clstrs, clstr_id


############################################################
# Get Threshold     
# - input PD: Persistence diagram
#         m: number of desired clusters must at least 1
#         multiplicative: whether death/birth (True) is used 
#                         or death - birth (False)
#
#         Note that if birth is 0 and multiplicative is used
#         an exception will be raised
# - returns: 
#        alpha: threshold
############################################################
def getThreshold(PD:np.array, m:int,multiplicative:bool=True)->float:
    # test that no vertices have value 0
    if multiplicative and (np.count_nonzero(PD[:,0])<PD.shape[0]):
        raise Exception("Cannot use multiplicative if vertex can take value 0")
    
    # compute persistence values

    if multiplicative:     
        pers_values = PD[:,1]/PD[:,0]
    else:
        pers_values = PD[:,1] - PD[:,0]

    pers_values = np.sort(pers_values)[::-1]        # sort in decreasing order

    return (pers_values[m]+pers_values[m-1])/2

############################################################
# Compute minimum spanning tree
# - input edges: list of tuples with (source,target,weight)
#        
# - returns: sublist corresponding to minimum spanning tree 
#            (or forest)
#         
############################################################
def computeMST(E:List[Tuple[int,int,float]])->List[Tuple[int,int,float]]:

    E.sort(key = lambda x: x[2])                                # the edges must be sorted
    
    # Initialization 
    verts = list(set([e[0] for e in E]+ [e[1] for e in E]))     # make list of vertices
    conn = dict(zip(verts,verts))                               # simplified union find 
                                                                # connected components look up table

    MST = []                                                    # initialize MST
    
    for x,y,f in E:
        #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # if they are the same then move on, otherwise
        if(x_source!=y_source):
             # apply lexicographical ordering (to prevent further tests)
             # NOTE: since these are representative vertices, we cannot 
             #       control the ordering a priori

            s = x_source if x_source<y_source else y_source     # We know they are difffernet here
            t = y_source if x_source<y_source else x_source 

            conn[t] = s # perform merge
            MST.append((x,y,f))
                        
    return MST



############################################################
# Compute persistence diagram for graph
# - input edges: list of tuples with (source,target,weight)
#         k: parameter for filtration
#            return_filtration: whether to return the edges filtration function 
#            (needed for clustering but with some overhead)
# - returns: if return_filtration is False
#                PD: numpy array with 1st column birth second column death
#            if return_filtration is True
#                PD: as above
#                F: numpy array with vertex filtration values
#                E: list of tuples with each tuple being one edge 
#                          (edge source, edge target, edge weight)  
# Notes: 1. this does not return points
#           on the diagonal,
#   
# TODO: Make wrapper functions for a cleaner pattern
############################################################
def persistenceDiagramGraph(edges:List[Tuple[int,int,float]],k:int, return_filtration:bool=False)->np.array | Tuple[np.array,np.array,List[Tuple[int,int,float]]]:
   

    edges.sort(key = lambda x: x[2])                            # make sure edges are sorted
    
    # Initialization 
    verts = list(set([e[0] for e in edges]+ [e[1] for e in edges]))     # make list of vertices
    conn = dict(zip(verts,verts))                               # simplified union find 
                                                                # connected components look up table

    weights = dict.fromkeys(verts,1)       # weight function
    F  = dict.fromkeys(verts,0)            # vertex filtration function
    PD = []                                # initialize empty persistence diagram

    if return_filtration:
        non_active = {v:set([v,]) for v in verts}   # to keep track of non-active 
                                                    # components to update filtration

    #------------------------------------------
    # Main loop - iterate over edges in order
    # - this is massively simplified since 
    #   all edges are negative 
    #------------------------------------------
    for x,y,f in edges:  # edges is (x,y) with edge weight f
       
        #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # apply lexicographical ordering (to prevent further tests)
        # NOTE: since these are representative vertices, we cannot 
        #       control the ordering a priori
        s = x_source if x_source<y_source else y_source
        t = y_source if x_source<y_source else x_source

        total_weight = weights[s]+weights[t]     # total weight of new component

        #---------------------------------------
        # Cases: 
        #---------------------------------------
        
        # neither component active
        if (weights[s]<k) and (weights[t]<k):
            # update weights to new root
            weights[s] = total_weight
            weights[t] = 0                               # for debugging purposes

            conn[t] = s                                  # update union find root
            
            if return_filtration:
                non_active[s].update(non_active[t])      # the pop empties the list and copies
                                                         # it into the root list for the component

            # if it has become active then record 
            # function of root vertex 
            if (weights[s]>=k):
                if return_filtration:                    # update entire component
                    for v in non_active[s]:
                        F[v] = f 
                else:
                     F[s] = f                            # update only root 

        # s component active, tnot
        elif (weights[s]>=k) and (weights[t]<k):
            weights[s] += weights[t]
            weights[t] = 0                               # for debugging purposes
            
            conn[t] = s                                  # update union find root
 
            if return_filtration:                        # update nonactive component
                for v in non_active[t]:
                    F[v] = f
              
        # t component active, s not (this is verbose but simple)
        elif (weights[s]<k) and (weights[t]>=k):
            weights[t] += weights[s]
            weights[s] = 0               # for debugging purposes
            
            conn[s] = t                  # update union find root

            if return_filtration:        # update nonactive component
                for v in non_active[s]:
                    F[v] = f

        # both components active
        elif (weights[s]>=k) and (weights[t]>=k):
            # F is defined here for both
            # redefine vertices wrt function value F
            w = s if F[s]<=F[t] else t
            v = t if F[s]<=F[t] else s
            
            weights[w] += weights[v]
            weights[v] = 0               # for debugging purposes
            
            conn[v] = w                  # update union find root 
            
            PD.append((F[v],f))          # add point to persistence diagram
        else:
            raise Exception("No case matches - bug")
    #------------------------------------------

    inf_bars = set([source(conn,i) for i in verts])   # find infinite components
    PD+=[(F[j],np.inf) for j in inf_bars]

    if return_filtration:
        E = [(e[0],e[1], max([e[2],F[e[0]],F[e[1]]])) for e in edges]
        E.sort(key = lambda x: x[2])
        return np.array(PD),F,E
    else:    
        return np.array(PD)
    

 ############################################################
# Get clusters - assumes persistence diagram computed from 
# the graph version
#     
# - input F: numpy array with filtration values for vertices
#         E: edges of MST with weights
#         alpha: threshold parameter
#         multiplicative: whether death/birth (True) is used 
#                         or death - birth (False)
#
#         Note that if birth is 0 and multiplicative is used
#         an exception will be raised
# - returns: 
#         clstrs: numpy array with representative vertex for 
#                 each cluster
#         clstr_id: a dictionary with a numbering for the 
#                   clusters so if you wanted the clusters
#                   named 1,2,3...
#                   [clstr_id[i] for i in clstrs]
############################################################
def getClustersGraph(F:Dict[int,float], E:Tuple[int,int,float], alpha:float, multiplicative:bool=True)->Tuple[np.array,Dict[int,int]]:
    
    
    # test that no vertices have value 0
    F_tmp = np.array([F[i] for i in F.keys()])
    if multiplicative and (np.count_nonzero(F_tmp)<F_tmp.shape[0]):
        raise Exception("Cannot use multiplicative if vertex can take value 0")
 
    verts = list(F.keys())                   # make list of vertices
    conn = dict(zip(verts,verts))            # simplified union find 

    E.sort(key = lambda x: x[2]) # re-sort edges just in case

    # main loop
    for x,y,f in E:
         #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # apply function and in case of equality lexicographical ordering
        # (to prevent further tests)
        # NOTE: since these are representative vertices, we cannot 
        #       control the ordering a priori
        s = x_source if comp((x_source,F[x_source]),(y_source,F[y_source])) else y_source
        t = y_source if comp((x_source,F[x_source]),(y_source,F[y_source])) else x_source

        # all edges are negative so only need to check threshold
        # whether to merge
        if multiplicative:
            if f/F[t]<alpha:
                conn[t]=s
        else:
            if f-F[t]<alpha:
                conn[t]=s

    # turn dictionary into np.array
    clstrs = np.zeros(len(verts))
    clstr_heads = set()
    for v in conn.keys():
        clstrs[v] = source(conn,v)
        clstr_heads.add(clstrs[v])
    
    clstr_id = dict(zip(clstr_heads,range(len(clstr_heads))))

    return clstrs, clstr_id