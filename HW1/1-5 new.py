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
import umap
from sklearn.manifold import TSNE
import plotly.express as px


G = nx.karate_club_graph()

communities = sorted(nx_comm.greedy_modularity_communities(G), key=len)
print(f"The club has {len(communities)} communities.")

for c, v_c in enumerate(communities):
    for v in v_c:
        G.nodes[v]['community'] = c + 1
        # add the new attribute(community number) to each node

label=[n[1]['community'] for n in G.nodes(data=True)]

G=nx.relabel_nodes(G, {n:str(n) for n in G.nodes()})

node2vec = Node2Vec(graph=G, dimensions=64, walk_length=30, p=0.4, q=0.8, num_walks=200, workers=1) 
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# model.wv.save_word2vec_format("emb.emb")

enbedlist=list(model.wv.key_to_index)
data=G.nodes(data=True)

labels=[]
for i in enbedlist:
    labels.append(data[i]['community'])

node_embeddings=model.wv.vectors

embedding=umap.UMAP(n_components=2).fit_transform(node_embeddings)

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=labels, 
    cmap=plt.cm.Accent)

plt.title('Karate Club UMAP Embedding', fontsize=12)
plt.show()
