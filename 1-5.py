# 2022-2 Machine learning with Graphs 2022711835 Junhee Kwon
# Homework 1
# 1.5

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

G=nx.relabel_nodes(G, {n:str(n) for n in G.nodes()})

# case 1 (initial value)
node2vec = Node2Vec(graph=G, dimensions=64, walk_length=30, p=0.5, q=0.5, num_walks=200, workers=1) 
model = node2vec.fit(window=10, min_count=1, batch_words=4)  

K = 3
kmeans = KMeans(n_clusters=K, random_state=0).fit(model.wv.vectors)
print(kmeans)

# print(model.wv.index_to_key)
# print(kmeans.labels_)

# node_label=[[n, label] for n, label in zip(model.wv.index_to_key, kmeans.labels_) ]
node_label=[[] for _ in range(K)]

for n, label in zip(model.wv.index_to_key, kmeans.labels_):
    node_label[label].append(n)
    
node_label2=sorted(node_label, key=len)

f=open("1-5", "w")
f.write("1-5\n\n")

for a in range(len(node_label2)):
    for b in node_label2[a]:
        G.nodes[b]['label'] = a+1

for a, data in sorted(G.nodes(data=True), key=lambda x: x[1]['label']):
    # print('{a} {w}'.format(a=a, w=data['label']))
    f.write('{a} {w} \n'.format(a=a, w=data['label']))

label=[n[1]['label'] for n in G.nodes(data=True)]

plt.figure(figsize=(12, 6))
nx.draw_networkx(G, pos=nx.layout.spring_layout(G), 
                 node_color=label, 
                 cmap=plt.cm.Accent
                )

plt.axis('off')
plt.show()