# 2022-2 Machine learning with Graphs 2022711835 Junhee Kwon
# Homework 1

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

G = nx.karate_club_graph()

# 1.4
graph=nx.karate_club_graph()
graph=nx.relabel_nodes(graph, {n:str(n) for n in graph.nodes()})

node2vec = Node2Vec(graph=graph, dimensions=64, walk_length=10, p=0.5, q=0.5, num_walks=200, workers=1) 
model = node2vec.fit(window=10, min_count=1, batch_words=4)  

K = 3
kmeans = KMeans(n_clusters=K, random_state=0).fit(model.wv.vectors)

for n, label in zip(model.wv.index_to_key, kmeans.labels_):
    graph.nodes[n]['label'] = label
label=[n[1]['label'] for n in graph.nodes(data=True)]
print(label)

plt.figure(figsize=(12, 6))
nx.draw_networkx(graph, pos=nx.layout.spring_layout(graph), 
                 node_color=label, 
                 cmap=plt.cm.rainbow
                )

plt.axis('off')
plt.show()

