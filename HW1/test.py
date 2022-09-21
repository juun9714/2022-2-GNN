from ge import deepwalk
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# load graph from networkx library
G = nx.karate_club_graph()

labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64)

# convert nodes from int to str format
keys = np.arange(0,34)
values = [str(i) for i in keys]
dic = dict(zip(keys, values))
H = nx.relabel_nodes(G, dic)

# train the model and generate embeddings
model = deepwalk(H, walk_length=10, num_walks=80, workers=1)
model.train(window_size=5,iter=3)

embeddings = model.get_embeddings()


# retrieve the labels for each node
labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64)

# assigning colours to node labels
color_map = []
for i in labels:
    if i == 0:
        color_map.append('blue')
    else: 
        color_map.append('red')  

# transform the embeddings from 128 dimensions to 2D space
m = TSNE(learning_rate=20, random_state=42)
tsne_features = m.fit_transform(list(embeddings.values()))

# plot the transformed embeddings
plt.figure(figsize=(9,6)) 
plt.scatter(x = tsne_features[:,0], 
            y = tsne_features[:,1],
            c = color_map,
            s =600,
            alpha=0.6)

# adds annotations
for i, label in enumerate(np.arange(0,34)):
    plt.annotate(label, (tsne_features[:,0][i], tsne_features[:,1][i]))

# save the visualization
plt.savefig('tsne.png', bbox_inches='tight',dpi = 1000)