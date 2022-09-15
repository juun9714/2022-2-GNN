# 2022-2 Machine learning with Graphs 2022711835 Junhee Kwon
# Homework 1

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm

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

plt.show()

# 각 edge가 연결하고 있는 두 노드의 id와, 해당 edge의 data를 반환
# print(G.edges(data=True))
# 각 edge가 연결하고 있는 두 노드의 id만 반환
# print(G.edges())