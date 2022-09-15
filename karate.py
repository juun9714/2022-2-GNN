# 2022-2 Machine learning with Graphs 2022711835 Junhee Kwon
# Homework 1

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

EMBEDDING_FILENAME="N2V"
EMBEDDING_MODEL_FILENAME="N2VModel"
EDGES_EMBEDDING_FILENAME="EN2V"
# 1.1
G = nx.karate_club_graph()

nx.draw_circular(G, with_labels=True)
plt.show()

# 1.2
print("Node Degree")
for v in G:
    print(f"{v:4} {G.degree(v):6}")


# 1.3

# Find the communities
communities = sorted(nx_comm.greedy_modularity_communities(G), key=len, reverse=True)
# Count the communities
print(f"The club has {len(communities)} communities.")

for c, v_c in enumerate(communities):
    for v in v_c:
        G.nodes[v]['community'] = c + 1
        # add the new attribute(community number) to each node

for a, b in G.edges():
    if G.nodes[a]['community']==G.nodes[b]['community']:
        G.edges[a,b]['community']=G.nodes[a]['community']
    else:
        G.edges[a,b]['community']=0


N_coms=len(communities)
edges_coms=[]
coms_G=[nx.Graph() for _ in range(N_coms)]

colors=['tab:blue', 'tab:orange', 'tab:green']
fig=plt.figure(figsize=(12,5))

for i in range(N_coms):
    edges_coms.append([(u,v,d) for u,v,d in G.edges(data=True) if d['community']==i+1])
    coms_G[i].add_edges_from(edges_coms[i])
    plt.subplot(1,3,i+1)
    plt.title('Community' + str(i+1))
    pos=nx.circular_layout(coms_G[i])
    nx.draw(coms_G[i], pos=pos, with_labels=True, node_color=colors[i])

print(coms_G[1])
plt.show()

# 각 edge가 연결하고 있는 두 노드의 id와, 해당 edge의 data를 반환
# print(G.edges(data=True))
# 각 edge가 연결하고 있는 두 노드의 id만 반환
# print(G.edges())


# 1.4
# Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
graph=nx.karate_club_graph()
graph=nx.relabel_nodes(graph, {n:str(n) for n in graph.nodes()})
#node, probability?

node2vec = Node2Vec(graph=graph, dimensions=64, walk_length=10, p=0.5, q=0.5, num_walks=200, workers=1) 
model = node2vec.fit(window=10, min_count=1, batch_words=4)  

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings
print(model.wv['2']) #embedding vector of node 2
print(model.wv.most_similar('2')) #10 similar nodes with the node 2

vectors_array = np.zeros((len(graph.nodes), 64))
for node in graph.nodes:
    vectors_array[int(node)] = model.wv[node]

# kmeans clustering 알고리즘 실행
kmeans = KMeans(n_clusters=5, random_state=0).fit(vectors_array)

#### 그래프 시각화 - 각 클러스터별로 다른 색깔을 갖도록 함 ####
pos = nx.spring_layout(graph)
node_color=[]
node_degree = []
for node in graph.nodes:
    node_degree.append(graph.degree[node]*10)
    i = int(node)
    if kmeans.labels_[i] == 0:
        node_color.append('red')
    elif kmeans.labels_[i] == 1:
        node_color.append('yellow')
    elif kmeans.labels_[i] == 2:
        node_color.append('blue')
    elif kmeans.labels_[i] == 3:
        node_color.append('green')
    else:
        node_color.append('orange')

img = nx.draw_networkx_nodes(graph, pos, node_color = node_color, node_size=node_degree)
plt.show()



# # Save embeddings for later use
# model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# # Save model for later use
# model.save(EMBEDDING_MODEL_FILENAME)

# # Embed edges using Hadamard method
# from node2vec.edges import HadamardEmbedder

# edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# # Look for embeddings on the fly - here we pass normal tuples
# edges_embs[('1', '2')]
# ''' OUTPUT
# array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
#        ... ... ....
#        ..................................................................],
#       dtype=float32)
# '''

# # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
# edges_kv = edges_embs.as_keyed_vectors()

# # Look for most similar edges - this time tuples must be sorted and as str
# edges_kv.most_similar(str(('1', '2')))

# # Save embeddings for later use
# edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)