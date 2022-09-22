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

node2vec = Node2Vec(graph=G, dimensions=64, walk_length=30, p=0.9999, q=0.00001, num_walks=200, workers=1) 
model = node2vec.fit(window=10, min_count=1, batch_words=4)

model.wv.save_word2vec_format("emb.emb")

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

# plt.text(embedding[:, 0]+0.1, embedding[:, 1], "point({}, {})".format(x, y), fontsize=10)
# plt.gca().set_aspect('equal', 'datalim')
plt.title('Karate Club UMAP Embedding', fontsize=12)
plt.show()





# K = 3
# kmeans = KMeans(n_clusters=K, random_state=0).fit(model.wv.vectors)

# node_label=[[] for _ in range(K)]

# for n, label in zip(model.wv.index_to_key, kmeans.labels_):
#     node_label[label].append(n)
    
# node_label2=sorted(node_label, key=len)

# for a in range(len(node_label2)):
#     for b in node_label2[a]:
#         G.nodes[b]['label'] = a+1

# label=[n[1]['label'] for n in G.nodes(data=True)]

# plt.figure(figsize=(12, 6))
# nx.draw_networkx(G, pos=nx.layout.spring_layout(G), 
#                  node_color=label, 
#                  cmap=plt.cm.Accent
#                 )

# plt.axis('off')
# plt.show()