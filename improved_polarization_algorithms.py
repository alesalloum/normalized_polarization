'''
    File name: improved_polarization_algorithms.py
    Author: Ali Salloum
    Date created: 01/09/2020
    Date last modified: 06/11/2023
    Python Version: 3.10.9
    Networkx Version: 3.2.1
    Scipy: 1.10.1
'''

import heapq
import math
import random

from typing import List, Tuple, Set, Any, Dict
from collections import Counter

import networkx as nx
import numpy as np
import scipy.stats

# Random Walk Controversy (RWC) & Adaptive Random Walk Controversy (ARWC)

def get_influencer_nodes(G: nx.Graph, left_nodes: List, right_nodes: List, n_influencers: float) -> Tuple[List, List]:
    """Returns the n_influencers highest-degree nodes for each partition of a graph.
    
    Args:
        G: A NetworkX graph.
        left_nodes: A list of nodes in the left partition.
        right_nodes: A list of nodes in the right partition.
        n_influencers: Number of top-degree nodes to find in each partition. Can be an integer
                       or a fraction of the total nodes.
    
    Returns:
        A tuple containing two lists: the top-degree nodes in the left and right partitions.
    """
    
    if not left_nodes or not right_nodes or n_influencers < 0:
        raise ValueError("Invalid input: nodes lists must be non-empty and n_influencers must be non-negative.")
    
    left_degrees = G.degree(left_nodes)
    right_degrees = G.degree(right_nodes)
    
    k_left = k_right = n_influencers
    if n_influencers < 1:
        k_left = max(1, int(n_influencers * len(left_nodes)))
        k_right = max(1, int(n_influencers * len(right_nodes)))
    
    left_influencers = heapq.nlargest(k_left, left_degrees, key=lambda x: x[1])
    right_influencers = heapq.nlargest(k_right, right_degrees, key=lambda x: x[1])
    
    return [node for node, _ in left_influencers], [node for node, _ in right_influencers]

def rwc_perform_random_walk(G: nx.Graph, left_influencers: Set[Any], right_influencers: Set[Any], starting_node: Any) -> str:
    """Performs a random walk on graph G starting from starting_node.
    
    The walk continues until it reaches a node in either left_influencers or right_influencers.
    Returns 'left' or 'right' depending on which set of influencers the walk ends on.
    
    Args:
        G: A NetworkX graph object.
        left_influencers: A set of nodes representing left influencers.
        right_influencers: A set of nodes representing right influencers.
        starting_node: The node from which the random walk begins.
        
    Returns:
        A string indicating the ending side of the walk ('left' or 'right').
    """
    # Convert to sets for efficient membership tests if not already sets
    left_influencers = set(left_influencers)
    right_influencers = set(right_influencers)
    
    # Check if the starting node is in G
    if starting_node not in G:
        raise ValueError("The starting node must be in the graph.")

    current_node = starting_node
    
    while True:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            raise ValueError("The current node has no neighbors to walk to.")
        
        next_node = random.choice(neighbors)
        
        if next_node in left_influencers:
            return "left"
        elif next_node in right_influencers:
            return "right"
        else:
            current_node = next_node

def RWC_polarization(G: nx.Graph, ms: Dict[Any, int], n_influencers: float, n_sim: int, n_walks: int) -> float:
    """Computes Random Walk Controversy Polarization.

    Args:
        G: A NetworkX graph object.
        ms: A dictionary with node as keys and side (0 or 1) as values.
        n_influencers: The number of influencer nodes.
        n_sim: The number of simulations to run.
        n_walks: The number of walks per simulation.
        
    Returns:
        The average RWC over all simulations.
    """
    left_nodes, right_nodes = [node for node in ms if ms[node] == 0], [node for node in ms if ms[node] == 1]
    
    left_influencers, right_influencers = get_influencer_nodes(G, left_nodes, right_nodes, n_influencers)
    
    rwc_dist = []

    for _ in range(n_sim):
        counts = Counter()

        for _ in range(n_walks):
            starting_side = random.choice(["left", "right"])
            which_random_starting_node = random.choice(left_nodes if starting_side == "left" else right_nodes)
            ending_side = rwc_perform_random_walk(G, left_influencers, right_influencers, which_random_starting_node)
            
            counts[starting_side + "_" + ending_side] += 1

        # Calculating RWC for each simulation
        left_left, right_left = counts['left_left'], counts['right_left']
        left_right, right_right = counts['left_right'], counts['right_right']

        e1, e2 = left_left / max(1, left_left + right_left), right_left / max(1, left_left + right_left)
        e3, e4 = left_right / max(1, right_right + left_right), right_right / max(1, right_right + left_right)
        
        rwc = e1 * e4 - e2 * e3
        rwc_dist.append(rwc)
    
    rwc_ave = sum(rwc_dist) / len(rwc_dist)
    
    return rwc_ave

# E-I Index (EI)

def EI_polarization(G: nx.Graph, ms: Dict[Any, int]) -> float:
    """Computes EI-Index Polarization.

    Args:
        G: A NetworkX graph object.
        ms: A dictionary with nodes as keys and assigned group (0 or 1) as values.
        
    Returns:
        The EI-Index indicating the polarization level of the graph.
    """
    EL = sum(1 for s, t in G.edges() if ms[s] != ms[t])
    IL = sum(1 for s, t in G.edges() if ms[s] == ms[t])
    
    total_links = EL + IL
    if total_links == 0:
        return 0  # Return zero to indicate no polarization for a graph with no edges.
    
    ei_score = -1 * ((EL - IL) / total_links)

    return ei_score

# Adaptive E-I Index (AEI)

def AEI_polarization(G: nx.Graph, ms: Dict[Any, int]) -> float:
    """Computes Adaptive EI-Index Polarization.

    Args:
        G: A NetworkX graph object.
        ms: A dictionary with nodes as keys and assigned group (0 or 1) as values.
        
    Returns:
        The Adaptive EI-Index indicating the polarization level of the graph.
    """
    block_a = {k for k, v in ms.items() if v == 0}
    block_b = {k for k, v in ms.items() if v == 1}

    n_a = len(block_a)
    n_b = len(block_b)

    # Counting internal connections using set intersections
    c_a = sum(1 for s, t in G.edges(block_a) if (t in block_a and s in block_a))
    c_b = sum(1 for s, t in G.edges(block_b) if (t in block_b and s in block_b))

    # Counting external connections
    c_ab = sum(1 for s, t in G.edges() if (s in block_a and t in block_b) or (s in block_b and t in block_a))

    # Density calculations with checks to prevent division by zero
    B_aa = (c_a / (n_a * (n_a - 1) / 2)) if n_a > 1 else 0
    B_bb = (c_b / (n_b * (n_b - 1) / 2)) if n_b > 1 else 0
    B_ab = (c_ab / (n_a * n_b)) if (n_a > 0 and n_b > 0) else 0

    # Since B_ba and B_ab are equivalent, no need to calculate B_ba separately
    B_ba = B_ab
        
    aei_score = (B_aa + B_bb - B_ab - B_ba) / (B_aa + B_bb + B_ab + B_ba)
    
    return aei_score

# Betweenness Centrality Controversy (BCC)

def BCC_polarization(G: nx.Graph, ms: Dict[Any, int]) -> float:
    """Computes Betweenness Centrality Controversy Polarization.

    Args:
        G: A NetworkX graph object.
        ms: A dictionary with nodes as keys and assigned group (0 or 1) as values.
        resample_size: The number of samples to draw in resampling.
        
    Returns:
        The average BCC over all simulations
    """
    dict_eb = nx.edge_betweenness_centrality(G, k=int(0.75*len(G)))

    cut_edges = [(s, t) for s, t in G.edges() if ms[s] != ms[t]]
    rest_edges = [(s, t) for s, t in G.edges() if ms[s] == ms[t]]

    if len(cut_edges) <= 1:
        raise ValueError("Not enough cut edges to compute the polarization.")

    cut_ebc = [dict_eb[e] for e in cut_edges]
    rest_ebc = [dict_eb[e] for e in rest_edges]

    kernel_for_cut = scipy.stats.gaussian_kde(cut_ebc, "silverman")
    kernel_for_rest = scipy.stats.gaussian_kde(rest_ebc, "silverman")

    bbc_dist = []
    n_iterations = 10
    resample_size = int(1e4)

    for _ in range(n_iterations):
        cut_dist = kernel_for_cut.resample(resample_size)[0]
        rest_dist = kernel_for_rest.resample(resample_size)[0]

        cut_dist = [max(1e-5, value) for value in cut_dist]
        rest_dist = [max(1e-5, value) for value in rest_dist]

        kl_divergence = scipy.stats.entropy(rest_dist, cut_dist)

        BCCval = 1 - math.e ** (-kl_divergence)
        
        bbc_dist.append(BCCval)
    
    bbc_score = sum(bbc_dist) / len(bbc_dist)

    return bbc_score

# Boundary Polarization (BP, GMCK)

def BP_polarization(G: nx.Graph, ms: Dict[Any, int]) -> float:
    """
    Computes the GMCK Polarization

    Args:
        G: A NetworkX graph object representing the network structure.
        ms: A dictionary mapping each node to its group membership, where 0 and 1
            denote different groups.

    Returns:
        The GMCK Polarization index as a float
    """

    X = set([node for node in ms if ms[node] == 0])
    Y = set([node for node in ms if ms[node] == 1])

    B = []

    for e in G.edges():
        s, t = e

        if ms[s] != ms[t]:
            B.extend([s,t])

    Bset = set(B)
    
    Bx = Bset & X
    By = Bset & Y    

    Ix = X.difference(Bx)
    Iy = Y.difference(By)
    
    I = Ix.union(Iy)

    Bxf = Bx.copy()
    Byf = By.copy()
    
    for u in Bxf:
        connections_of_u = set([n for n in G.neighbors(u)])
        if connections_of_u.issubset(B):
            Bx.remove(u)

    for v in Byf:
        connections_of_v = set([n for n in G.neighbors(v)])
        if connections_of_v.issubset(B):
            By.remove(v)

    B = Bx.union(By)

    summand = []
    for node in B:

        di = nx.cut_size(G, [node], I)
        
        if node in Bx:
            db = nx.cut_size(G, [node], Byf)
            
        else:
            db = nx.cut_size(G, [node], Bxf)
            
        summand.append((di/(di+db)) - 0.5)

    gmck_score = (1/(len(B)+0.0001)) * sum(summand)

    return gmck_score

# Dipole Polarization (DP, MBLB)

def DP_polarization(G: nx.Graph, ms: Dict[Any, int], n_influencers: float) -> float:
    """
    Computes the DP (Dipole Polarization) index for a network graph based on
    the distribution of polarity among its nodes. Influential nodes are identified
    and assigned a polarity, and this polarity is then diffused throughout the
    network to all other nodes.

    Args:
        G: A NetworkX graph object representing the network structure.
        ms: A dictionary mapping each node to its group membership, where 0 and 1
            denote different groups.

    Returns:
        The DP Polarization index as a float
    """

    left_nodes = [node for node in ms if ms[node] == 0]
    right_nodes = [node for node in ms if ms[node] == 1]
    
    X_top, Y_top = get_influencer_nodes(G, left_nodes, right_nodes, n_influencers)
    
    X_top = set(X_top)
    Y_top = set(Y_top)

    dict_polarity = dict.fromkeys(list(G), 0)

    dict_polarity.update(zip(X_top, [-1] * len(X_top)))
    dict_polarity.update(zip(Y_top, [1] * len(Y_top)))

    listeners = set(G.nodes) - X_top - Y_top
    
    polarity = np.asarray(list(dict_polarity.values()))
    polarity_new = np.zeros(len(polarity))

    roundcount = 0
    tol=10**-5
    
    notconverged = len(polarity_new)
    max_rounds = 500

    while notconverged > 0 :

        for node in listeners:

            polarity_new[node] = np.mean([polarity[n] for n in G.neighbors(node)])

        if roundcount < 1:
            polarity_new[list(X_top)] = -1
            polarity_new[list(Y_top)] = 1

        diff = np.abs(polarity - polarity_new)
        notconverged = len(diff[diff>tol])

        polarity = polarity_new.copy()

        if roundcount > max_rounds:

            #print("Maximum rounds achieved")
            break

        roundcount += 1

    #print("Rounds needed: ", roundcount)
    
    n_nodes = len(G)
    n_plus = len(polarity[polarity > 0]) 
    n_minus = n_nodes - n_plus

    delta_A = np.abs((n_plus - n_minus) * (1/(n_nodes)))

    gc_plus = np.mean(polarity[polarity > 0])
    gc_minus = np.mean(polarity[polarity < 0])

    pole_D = np.abs(gc_plus-gc_minus) * (1/2)
    mblb_score = (1-delta_A) * pole_D
    
    return mblb_score

# Modularity based Polarization (Q)

def Q_polarization(G: nx.Graph, ms: Dict[Any, int]) -> float:
    """
    Computes the modularity score for a graph based on a given node partitioning. 

    Args:
        G: A NetworkX graph object representing the network structure.
        ms: A dictionary mapping each node to its group membership, where nodes 
            labeled '0' are in one group and nodes labeled '1' are in another.

    Returns:
        The modularity score as a float
    """

    c1_metis = [node for node in ms if ms[node] == 0]
    c2_metis = [node for node in ms if ms[node] == 1]
    
    mod_score = nx.community.modularity(G, [c1_metis, c2_metis])

    return mod_score
