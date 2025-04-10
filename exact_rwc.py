import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

def markov_rwc(
    G: nx.Graph,
    ms_partition: Dict[int, int],
    rwc_n_influencers: int,
    get_influencers_fn=None
) -> float:
    """
    Compute the Random Walk Controversy (RWC) score for a graph using an exact Markov chain formulation.

    Parameters:
        G (nx.Graph): The input graph.
        ms_partition (Dict[int, int]): A mapping from node to community (0 or 1).
        rwc_n_influencers (int): Number of influencers (absorbing states) to choose per community.
        get_influencers_fn (Callable): A function to extract influencers, signature:
            (G, rwc_n_influencers, ms_partition) -> Tuple[List[int], List[int]]

    Returns:
        float: The computed RWC score.
    """
    if get_influencers_fn is None:
        raise ValueError("Please provide a valid 'get_influencers_fn' function.")

    left_nodes = [node for node, label in ms_partition.items() if label == 0]
    right_nodes = [node for node, label in ms_partition.items() if label == 1]

    left_absorbers, right_absorbers = get_influencers_fn(G, rwc_n_influencers, ms_partition)

    all_absorbers = set(left_absorbers + right_absorbers)
    transients = list(set(G.nodes) - all_absorbers)
    nodelist = transients + left_absorbers + right_absorbers

    # Mapping for reordered matrix
    mapping = {node: idx for idx, node in enumerate(nodelist)}

    left_mask = [mapping[node] for node in left_nodes if node not in all_absorbers]
    right_mask = [mapping[node] for node in right_nodes if node not in all_absorbers]

    # Construct transition matrix
    A = nx.to_numpy_array(G, nodelist=nodelist)
    degree_sums = A.sum(axis=1)
    degree_sums[degree_sums == 0] = 1  # avoid division by zero
    D_inv = np.diag(1 / degree_sums)
    P = D_inv @ A  # transition probability matrix

    T = len(transients)
    R = 2 * rwc_n_influencers

    # Canonical form: top-left (Q), top-right (0), bottom-left (R), bottom-right (I)
    zero_block = np.zeros((R, T))
    identity_block = np.eye(R)
    absorbing_chunk = np.hstack((zero_block, identity_block))
    canonical_matrix = np.vstack((P[:T, :], absorbing_chunk))

    # Fundamental matrix
    N = np.linalg.inv(np.eye(T) - canonical_matrix[:T, :T])
    B = N @ canonical_matrix[:T, T:]  # Absorbing probabilities

    # Probabilities
    own_bubble_L = B[left_mask, :rwc_n_influencers]
    cross_bubble_L = B[left_mask, rwc_n_influencers:]

    own_bubble_R = B[right_mask, rwc_n_influencers:]
    cross_bubble_R = B[right_mask, :rwc_n_influencers]

    PXX = np.mean(own_bubble_L.sum(axis=1))
    PYY = np.mean(own_bubble_R.sum(axis=1))
    PXY = np.mean(cross_bubble_L.sum(axis=1))
    PYX = np.mean(cross_bubble_R.sum(axis=1))

    rwc_score = PXX * PYY - PXY * PYX
    return float(rwc_score)
