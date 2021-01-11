import random

import scipy
import scipy.sparse
from scipy.sparse.linalg import eigsh
import networkx as nx
import numpy as np 
import metis

def regularized_laplacian_matrix(adj_matrix, tau):
    """
    The original code for regularized spectral clustering was written by samialabed in 2018. We modified the code for our
    purposes. Original script: https://github.com/samialabed/regualirsed-spectral-clustering
    
    Using ARPACK solver, compute the first K eigen vector.
    The laplacian is computed using the regularised formula from [2]
    [2]Kamalika Chaudhuri, Fan Chung, and Alexander Tsiatas 2018.
        Spectral clustering of graphs with general degrees in the extended planted partition model.

    L = I - D^-1/2 * A * D ^-1/2

    :param adj_matrix: adjacency matrix representation of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param tau: the regularisation constant
    :return: the first K eigenvector
    """
    # Code inspired from nx.normalized_laplacian_matrix, with changes to allow regularisation
    n, m = adj_matrix.shape
    #I = np.eye(n, m)
    I = scipy.sparse.identity(n, dtype='int8', format='dia')
    diags = adj_matrix.sum(axis=1).flatten()
    # add tau to the diags to produce a regularised diags
    if tau != 0:
        diags = np.add(diags, tau)

    # diags will be zero at points where there is no edge and/or the node you are at
    #  ignore the error and make it zero later
    with scipy.errstate(divide='ignore'):
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    D = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')

    L = I - (D.dot(adj_matrix.dot(D)))
    
    return L

def eigen_solver(laplacian, n_clusters):
    """
    ARPACK eigen solver in Shift-Invert Mode based on http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    """
    lap = laplacian * -1
    v0 = np.random.uniform(-1, 1, lap.shape[0])
    eigen_values, eigen_vectors = eigsh(lap, k=n_clusters, sigma=1.0, v0=v0)
    eigen_vectors = eigen_vectors.T[n_clusters::-1]
    
    return eigen_values, eigen_vectors[:n_clusters].T


def regularized_spectral_clustering(adj_matrix, tau, n_clusters, algo='scan'):
    """
    :param adj_matrix: adjacency matrix representation of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param n_clusters: cluster partitioning constant
    :param algo: the clustering separation algorithm, possible value kmeans++ or scan
    :return: labels, number of clustering iterations needed, smallest set of cluster found, execution time
    """
    regularized_laplacian = regularized_laplacian_matrix(adj_matrix, tau)
    eigen_values, eigen_vectors = eigen_solver(regularized_laplacian, n_clusters=n_clusters)

    if n_clusters == 2:  # cluster based on sign
        second_eigen_vector_index = np.argsort(eigen_values)[1]
        second_eigen_vector = eigen_vectors.T[second_eigen_vector_index]
        labels = [0 if val <= 0 else 1 for val in second_eigen_vector]  # use only the second eigenvector
        num_iterations = 1
    
    return labels

def evaluate_graph(graph, n_clusters):
    """
    Reconsutrction of [1]Understanding Regularized Spectral Clustering via Graph Conductance, Yilin Zhang, Karl Rohe

    :param graph: Graph to be evaluated
    :param n_clusters: How many clusters to look at
    :param graph_name: the graph name used to create checkpoints and figures
    :return:
    """
    graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

    graph_degree = graph.degree()
    graph_average_degree = np.sum(val for (node, val) in graph_degree) / graph.number_of_nodes()

    adj_matrix = nx.to_scipy_sparse_matrix(graph, format='csr')
    tau = graph_average_degree

    labels = regularized_spectral_clustering(adj_matrix, tau, n_clusters, 'scan')
    
    return labels

def mod_partition(G, ms):

    T = set([k for k in ms if ms[k] == 0])
    S = set([k for k in ms if ms[k] == 1])
    
    current_modularity = nx_comm.modularity(G, [T,S])

    nodes = list(G.nodes)
    
    for _ in range(2):

        random.shuffle(nodes)

        for ranNode in nodes:
            xT = T.copy()
            xS = S.copy()

            if ranNode in S:
                xT.add(ranNode)
                xS.remove(ranNode)
            else:
                xT.remove(ranNode)
                xS.add(ranNode)

            proposed_modularity = nx_comm.modularity(G, [xT,xS])
            delta_Q = proposed_modularity - current_modularity

            if delta_Q > 0:

                T, S = xT, xS
                current_modularity = proposed_modularity
                
    ms_maxmod = dict()
    
    for n in G.nodes:
        if n in T:
            ms_maxmod[n] = 0
        else:
            ms_maxmod[n] = 1

    return ms_maxmod

def partition_spectral(G):
    
    n_partitions = 2
    labels = evaluate_graph(G, n_partitions)
    node_membership = dict(zip(G.nodes, labels))
    
    return node_membership

def partition_metis(G, ufac=400):
    
    adjList = [tuple(nbrdict.keys()) for _, nbrdict in G.adjacency()]
    _, parts = metis.part_graph(adjList, 2, ufactor=ufac)
    node_membership = dict(zip(G.nodes, parts))
    
    return node_membership

def partition_maxmod(G):
    
    initial_ms = partition_metis(G)
    node_membership = mod_partition(G, initial_ms)
    
    return node_membership
