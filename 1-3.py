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
print(G.nodes(data=True))
# Find the communities
communities = sorted(nx_comm.greedy_modularity_communities(G), key=len, reverse=True)
# Count the communities
print(f"The club has {len(communities)} communities.")

for c, v_c in enumerate(communities):
    print(c, v_c)
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

# for a, b in G.edges():
#     if G.nodes[a]['community']==G.nodes[b]['community']:
#         G.edges[a,b]['community']=G.nodes[a]['community']
#     else:
#         G.edges[a,b]['community']=0


# N_coms=len(communities)
# edges_coms=[]
# coms_G=[nx.Graph() for _ in range(N_coms)]

# colors=['tab:blue', 'tab:orange', 'tab:green']

# fig=plt.figure(figsize=(12,5))

# for i in range(N_coms):

#     edges_coms.append([(u,v,d) for u,v,d in G.edges(data=True) if d['community']==i+1])
#     coms_G[i].add_edges_from(edges_coms[i])
#     plt.subplot(1,3,i+1)
#     plt.title('Community' + str(i+1))
#     pos=nx.circular_layout(coms_G[i])
#     nx.draw(coms_G[i], pos=pos, with_labels=True, node_color=colors[i])

# plt.show()