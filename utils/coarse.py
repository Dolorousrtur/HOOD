"""
Tools for constructing sets of long-range (coarse) edges for a mesh
"""

import networkx as nx
import numpy as np
import torch

from utils.common import triangles_to_edges



def make_graph_from_faces(faces):
    faces = torch.LongTensor(faces)
    senders, receivers = triangles_to_edges(faces[None])
    edges = torch.stack([senders, receivers], dim=-1).numpy()

    G = nx.Graph()
    G.add_nodes_from(np.arange(faces.max() + 1))
    G.add_edges_from(edges)
    return G

def make_graph_from_edges(edges):
    G = nx.Graph()
    G.add_nodes_from(np.unique(edges.reshape(-1)))
    G.add_edges_from(edges)
    return G


def make_distance_row(G, source):
    N = np.array(G.nodes).shape[0]
    row = np.zeros(N)

    distance_dict = nx.shortest_path_length(G, source=source)
    inds = np.array(list(distance_dict.keys())).astype(np.int32)
    vals = np.array(list(distance_dict.values()))

    row[inds] = vals

    return row


def make_subsampled_graph(G, center):
    distances2center = make_distance_row(G, center).astype(np.int32)

    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes())

    for i in np.array(G.nodes):
        d2c = distances2center[i]
        if d2c % 2 == 0:
            continue

        adj2current = np.array(list(nx.neighbors(G, i)), dtype=np.int32)
        adj_closer = adj2current[distances2center[adj2current] == (d2c - 1)]
        adj_farther = adj2current[distances2center[adj2current] == (d2c + 1)]

        for ac in adj_closer:
            for af in adj_farther:
                if ac != af:
                    dist = nx.shortest_path_length(G, source=ac, target=af)
                    if dist == 2:
                        G2.add_edge(ac, af)

    G3 = nx.Graph()
    G3.add_nodes_from(G2.nodes())
    G3.add_edges_from(G2.edges())

    for i in np.array(G.nodes):
        d2c = distances2center[i]
        adj2current = np.array(list(nx.neighbors(G, i)), dtype=np.int32)
        if d2c % 2 == 0:
            continue
        adj_closer = adj2current[distances2center[adj2current] == (d2c - 1)]
        adj_farther = adj2current[distances2center[adj2current] == (d2c + 1)]

        if adj_farther.shape[0]:
            continue
        else:
            adj_farther = adj_closer

        for ac in adj_closer:
            for af in adj_farther:
                if ac != af:
                    dist2 = nx.shortest_path_length(G2, source=ac, target=af)
                    if dist2 > 2:
                        G3.add_edge(ac, af)

    for i in np.array(G.nodes):
        d2c = distances2center[i]
        adj2current = np.array(list(nx.neighbors(G, i)), dtype=np.int32)
        if d2c % 2 == 0:
            adj_closer = adj2current[distances2center[adj2current] % 2 == 0]
            adj_farther = np.array([i])
        else:
            continue

        for ac in adj_closer:
            for af in adj_farther:
                dist2 = nx.shortest_path_length(G3, source=ac, target=af)
                if dist2 > 2:
                    G3.add_edge(ac, af)

    return G3


def make_coarse_edges(faces, center, n_levels=1):
    """
    Construct several levels of coarse edges

    :param faces: [Fx3] numpy array of faces
    :param center: index od a center node in the mesh (see center of a graph)
    :param n_levels: number of long-range levels to construct
    :return: dictionary of long-range edges for each level
    """
    out_dict = {}
    G = make_graph_from_faces(faces)

    for i in range(n_levels):
        G = make_subsampled_graph(G, center)

        edges = np.array(G.edges)
        out_dict[i] = edges

    return out_dict
