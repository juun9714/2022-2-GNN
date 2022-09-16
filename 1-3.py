# 2022-2 Machine learning with Graphs 2022711835 Junhee Kwon
# Homework 1
# 1.3

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt


G = nx.karate_club_graph()
communities = sorted(nx_comm.greedy_modularity_communities(G), key=len, reverse=True)
print(f"The club has {len(communities)} communities.")

for c, v_c in enumerate(communities):
    for v in v_c:
        G.nodes[v]['community'] = c + 1
        # add the new attribute(community number) to each node

plt.figure(figsize=(12, 6))
nx.draw_networkx(G, pos=nx.layout.spring_layout(G), 
                 node_color=[n[1]['community'] for n in G.nodes(data=True)], 
                 cmap=plt.cm.rainbow
                )
                
plt.axis('off')
plt.show()
