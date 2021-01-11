'''
    File name: dk_analysis.py
    Author: Ali Salloum
    Date created: 01/09/2020
    Date last modified: 11/01/2021
    Python Version: 3.6
'''
import random
import pickle
import sys

import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
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

def zerok(R, n_samples):
    
    n, m = len(R.nodes), len(R.edges)
    
    buffer = []
    for i in range(n_samples):
        print("Processing sample number: ", i+1)
        G = nx.gnm_random_graph(n, m)
        buffer.append(compute_polarization(G))
        
    results_averaged = np.mean(buffer, axis=0)
    results_errors = np.std(buffer, axis=0)
    
    return [results_averaged, results_errors]

def onek(R, n_samples):
    
    degree_sequence = [d for v, d in R.degree()]
    
    buffer = []
    for i in range(n_samples):
        print("Processing sample number: ", i+1)
        G = nx.Graph(nx.configuration_model(degree_sequence))
        G.remove_edges_from(nx.selfloop_edges(G))
        buffer.append(compute_polarization(G))
        
    results_averaged = np.mean(buffer, axis=0)
    results_errors = np.std(buffer, axis=0)
    
    return [results_averaged, results_errors]

def twok(R, n_samples):
    
    R.remove_edges_from(nx.selfloop_edges(R))
    degree_sequence = [d for v, d in R.degree()]
    deg_dict = dict(zip(R.nodes(), degree_sequence))

    nx.set_node_attributes(R, deg_dict, "degree")
    joint_degrees = nx.attribute_mixing_dict(R, "degree")
    
    buffer = []
    for i in range(n_samples):
        print("Processing sample number: ", i+1)
        G = nx.joint_degree_graph(joint_degrees)
        buffer.append(compute_polarization(G))
        
    results_averaged = np.mean(buffer, axis=0)
    results_errors = np.std(buffer, axis=0)
    
    return [results_averaged, results_errors]

if __name__=="__main__":
    
    args = sys.argv
    net_name = sys.argv[1]
    
    n_samples = 5
   
    G = pickle.load(open("data/" + net_name, "rb"))
        
    RESULTS = dict()
    
    print("Processing network of: ", net_name)
    print("NK")
    print("***********************************")
    OBS_RES = compute_polarization(G)
    print("0K")
    print("***********************************")
    ZEROK_RES = zerok(G, n_samples)
    print("1K")
    print("***********************************")
    ONEK_RES = onek(G, n_samples)
    print("2K")
    print("***********************************")
    TWOK_RES = twok(G, n_samples)

    RESULTS["oG"] = G
    RESULTS["obs"] = OBS_RES
    RESULTS["0k"] = ZEROK_RES
    RESULTS["1k"] = ONEK_RES
    RESULTS["2k"] = TWOK_RES
        
    pickle.dump(RESULTS, open('finalresults2/infopack_' + str(net_name), 'wb'))
