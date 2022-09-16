# 2022-2 Machine learning with Graphs 2022711835 Junhee Kwon
# Homework 1
# 1.2

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

# 1.2
G = nx.karate_club_graph()

print("Node Degree")
for v in G:
    print(f"{v:4} {G.degree(v):6}")