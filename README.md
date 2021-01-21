# Separating Controversy from Noise: Comparison and Normalization of Structural Polarization Measures
The data and scripts for "Separating Controversy from Noise: Comparison and Normalization of Structural Polarization Measures" -paper

## Data

The network_data contains two subfolders (single_hashtag_networks and multiple_hashtag_networks). All the networks are in .edgelist format. The folder contains 183 topic endorsement networks inferred from Twitter interactions during the 2019 Finnish Elections. Nodes are accounts and undirected ties indicate uni- or bi-directional endorsement via retweets on the given topic. Please see [1] and [2] for details. No identifying information nor original raw data from the Twitter platform is included here. See the *network_info.csv* for more details on each network.

## Scripts

*polarization_algorithms.py*: the implementations of all polarization measures analyzed in the paper

*partition_algorithms.py*: the different partition algorithms used for obtaining the two groups

*dk_analysis.py*: computing the polarization scores for randomized networks

*deghet_analysis.py*: the analysis of non-homogeneous degree sequences on polarization scores

## References
<a id="1">[1]</a> 
Chen, Ted Hsuan Yun, et al. "Polarization of Climate Politics Results from Partisan Sorting: Evidence from Finnish Twittersphere." arXiv preprint arXiv:2007.02706 (2020).

[2] Salloum, Ali, et al. "Separating Controversy from Noise: Comparison and Normalization of Structural Polarization Measures." arXiv preprint arXiv:2101.07009
