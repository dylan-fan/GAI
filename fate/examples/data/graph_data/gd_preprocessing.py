import pandas as pd
import numpy as np
from collections import defaultdict

def split_edges(file, directed=False, frac=0.5, sep=' ', header=None):
    """
    Split the whole graph to two parts
    Parameters:
    ----------
    frac: The fraction of the first part w.r.t the whole graph
    """
    edges = pd.read_csv(file, sep, header=header)
    edges.columns = ['node1', 'node2']
    print(edges.head())

    edges_num = edges.shape[0]
    print("The number of edges in total:{}".format(edges_num))

    indices = [_ for _ in range(edges_num)]
    np.random.shuffle(indices)
    
    separate_point = int(edges_num * frac)
    part_one_edges = edges.iloc[indices[:separate_point], :]
    part_two_edges = edges.iloc[indices[separate_point:], :]

    print("The number of part one edges:{}".format(part_one_edges.shape[0]))
    print("The number of part one edges:{}".format(part_two_edges.shape[0]))

    # if the graph is undirected
    if directed == False:
        part_one_reverse_edges = pd.DataFrame({'node1': part_one_edges['node2'].values, 'node2': part_one_edges['node1'].values})
        part_one_edges = part_one_edges.append(part_one_reverse_edges)

        part_two_reverse_edges = pd.DataFrame({'node1': part_two_edges['node2'].values, 'node2': part_two_edges['node1'].values})
        part_two_edges = part_two_edges.append(part_two_reverse_edges)

    part_one_edges.to_csv('karate_edges_a.csv', index=False)
    part_two_edges.to_csv('karate_edges_b.csv', index=False)

    part_one_nodes = set(part_one_edges['node1']) | set(part_one_edges['node2'])
    part_two_nodes = set(part_two_edges['node1']) | set(part_two_edges['node2'])

    totoal_nodes = part_one_nodes | part_two_nodes
    common_nodes = part_one_nodes & part_two_nodes

    print("The nodes' number of part one:{}".format(len(part_one_nodes)))
    print("The nodes' number of part two:{}".format(len(part_two_nodes)))
    print("The total nodes' number of part one and two:{}".format(len(totoal_nodes)))
    print("The number of common nodes between part one and two:{}".format(len(common_nodes)))

def edge_to_adjlist(file_a, file_b):
    part_one_edges = pd.read_csv(file_a)
    print(part_one_edges.shape)
    part_two_edges = pd.read_csv(file_b)
    print(part_two_edges.head())

    def df_to_adj(df):
        graph = defaultdict(set)
        for row in range(df.shape[0]):
            nodes = tuple(df.iloc[row, :])
            if nodes[0] != nodes[1]:
                graph[nodes[0]].add(nodes[1])
        
        return graph

    def to_csv(graph, file):
        f = open(file, 'w')
        f.write("Nodes Adjlist\n")
        for key in graph.keys():
            adj_list = graph[key]
            line = str(key) + ','
            line += ','.join([str(_) for _ in adj_list])
            f.write(line + '\n')
        f.close()

    graph_one = df_to_adj(part_one_edges)
    graph_two = df_to_adj(part_two_edges)

    print("The part one has {} nodes".format(len(graph_one.keys())))
    print("The part one has {} nodes".format(len(graph_two.keys())))
    common_nodes = set(graph_one.keys()) & set(graph_two.keys())
    print("The part one has {} nodes".format(len(common_nodes)))

    to_csv(graph_one, '/fate/examples/data/karate_a.csv')
    to_csv(graph_two, '/fate/examples/data/karate_b.csv')



file = r'/fate/examples/data/graph_data/karate/karate.edgelist'
split_edges(file)
edge_to_adjlist('/fate/karate_edges_a.csv', '/fate/karate_edges_b.csv')
#data = pd.read_csv('citeseer_a.csv')
#print(len(set(data['id'])))

#data = pd.read_csv('citeseer_b.csv')
#print(len(set(data['id'])))
