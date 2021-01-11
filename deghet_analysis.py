'''
    File name: deghet_analysis.py
    Author: Ali Salloum
    Date created: 01/09/2020
    Date last modified: 11/01/2021
    Python Version: 3.6
'''
import sys
import random
import pickle

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import scipy.stats

import partition_algorithms as pa
import polarization_algorithms as pol

def get_giant_component(dG):
    Gcc = sorted(nx.connected_components(dG), key=len, reverse=True)
    G_Giant = dG.subgraph(Gcc[0])
    G = nx.convert_node_labels_to_integers(G_Giant)
    
    return G

def get_average_degree(dG):
    degree_sequence = [d for n,d in dG.degree()]
    average_degree = sum(degree_sequence)/len(degree_sequence)
    
    return average_degree

def compute_polarization(dG):
    G = get_giant_component(dG)

    ms_rsc = pa.partition_spectral(G)
    T_rsc = [node for node in ms_rsc if ms_rsc[node] == 0]
    S_rsc = [node for node in ms_rsc if ms_rsc[node] == 1]
    print("RSC completed.")
    
    ms_metis = pa.partition_metis(G)
    T_metis = [node for node in ms_metis if ms_metis[node] == 0]
    S_metis = [node for node in ms_metis if ms_metis[node] == 1]
    print("METIS completed.")

    n_sim, n_walks = 5, int(1e4)
    
    rwc_rsc = pol.random_walk_pol(G, ms_rsc, 10, n_sim, n_walks)
    rwc_metis = pol.random_walk_pol(G, ms_metis, 10, n_sim, n_walks)
    print("RWC nonadaptive completed.")
    
    arwc_rsc = pol.random_walk_pol(G, ms_rsc, 0.01, n_sim, n_walks)
    arwc_metis = pol.random_walk_pol(G, ms_metis, 0.01, n_sim, n_walks)
    print("RWC adaptive completed.")
    
    cond_rsc = 1-nx.conductance(G, S_rsc, T_rsc)
    cond_metis = 1-nx.conductance(G, S_metis, T_metis)
    print("Conductance completed.")
    
    mod_rsc = nx_comm.modularity(G, [S_rsc, T_rsc])
    mod_metis = nx_comm.modularity(G, [S_metis, T_metis])
    print("Modularity completed.")
    
    ei_rsc = -1*pol.krackhardt_ratio_pol(G, ms_rsc)
    ei_metis = -1*pol.krackhardt_ratio_pol(G, ms_metis)
    print("E-I completed.")
    
    extei_rsc = -1*pol.extended_krackhardt_ratio_pol(G, ms_rsc)
    extei_metis = -1*pol.extended_krackhardt_ratio_pol(G, ms_metis)
    print("Extended E-I completed.")
    
    ebc_rsc = pol.betweenness_pol(G, ms_rsc)
    ebc_metis = pol.betweenness_pol(G, ms_metis)
    print("BCC completed.")
    
    gmck_rsc = pol.gmck_pol(G, ms_rsc)
    gmck_metis = pol.gmck_pol(G, ms_metis)
    print("GMCK completed.")
    
    mblb_rsc = pol.dipole_pol(G, ms_rsc)
    mblb_metis = pol.dipole_pol(G, ms_metis)
    print("MBLB completed.")
    
    ave_deg = get_average_degree(G)
    size = len(G)
    
    infopack = [rwc_metis, arwc_metis, ebc_metis, gmck_metis, mblb_metis, mod_metis, ei_metis, extei_metis, cond_metis, 
                rwc_rsc, arwc_rsc, ebc_rsc, gmck_rsc, mblb_rsc, mod_rsc, ei_rsc, extei_rsc, cond_rsc, 
                size, ave_deg]
    
    return infopack

# n = int(1e5)
# n=50000

args = sys.argv

n = int(args[1])
prw_exp = float(args[2])
n_samples = 20

epsilon = float(1e-2)
gamma = prw_exp + epsilon
m = n ** .6 
#m = n/2
p = 1/(gamma-1)

results = []
print("Processing power-law exponent. ", gamma-epsilon)
average_degree_list = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
for average_degree in average_degree_list:
    print("Processing average degree of: ", average_degree)
    
    d = average_degree
    c = (1-p)*d*(n**p)
    i0 = (c/m)**(1/p) - 1
    w = [c/((i+i0)**p) for i in range(1,n+1)]
    
    for idx in range(n_samples):
        print("Processing sample: ", idx+1)
        
        G_nx = nx.expected_degree_graph(w)
        Giant_G_nx = get_giant_component(G_nx)
        infopack = compute_polarization(Giant_G_nx)
        results.append(infopack)

pickle.dump(results, open("results2/" + str(prw_exp) + "_" + str(n) + "_CM.p", "wb"))
