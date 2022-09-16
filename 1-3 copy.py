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
communities = nx_comm.greedy_modularity_communities(G)
print(communities)
print(sorted(communities))
print(sorted(communities,key=len))
print(f"The club has {len(communities)} communities.")

for c, v_c in enumerate(communities):
    for v in v_c:
        G.nodes[v]['community'] = c + 1
        # add the new attribute(community number) to each node

f=open("1-3", "w")
f.write("1-3\n\n")


for a, data in sorted(G.nodes(data=True), key=lambda x: x[1]['community']):
    print('{a} {w}'.format(a=a, w=data['community']))
    f.write('{a} {w} \n'.format(a=a, w=data['community']))
    
label=[n[1]['community'] for n in G.nodes(data=True)]

plt.figure(figsize=(12, 6))
nx.draw_networkx(G, pos=nx.layout.spring_layout(G), 
                 node_color=label, 
                 cmap=plt.cm.Accent
                )
                
plt.axis('off')
plt.show()