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
Chen, Ted Hsuan Yun ; Salloum, Ali ; Gronow, Antti ; Ylä-Anttila, Tuomas ; Kivelä, Mikko. / Polarization of climate politics results from partisan sorting : Evidence from Finnish Twittersphere. In: Global Environmental Change. 2021 ; Vol. 71. (https://doi.org/10.1016/j.gloenvcha.2021.102348)

<a id="2">[2]</a> Salloum, Ali ; Chen, Ted Hsuan Yun ; Kivelä, Mikko. / Separating Polarization from Noise: Comparison and Normalization of Structural Polarization Measures. In: Proc. ACM Hum.-Comput. Interact. 6, CSCW1 ; 2022 ; Article 115 ; Vol 6. (https://doi.org/10.1145/3512962)

*The network data is from the first source, and the polarization scores from the latter*
