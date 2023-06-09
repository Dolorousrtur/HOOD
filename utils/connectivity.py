import numpy as np
import torch
from sklearn import neighbors


def compute_connectivity_np(positions_from, positions_to, radius, k=None):
    """Get the indices of connected edges with radius connectivity.
    Args:
      positions: Positions of nodes in the graph. Shape:
        [num_nodes_in_graph, num_dims].
      radius: Radius of connectivity.
      add_self_edges: Whether to include self edges or not.
    Returns:
      indices_from indices [num_edges_in_graph]
      indices_to indices [num_edges_in_graph]
    """
    tree = neighbors.KDTree(positions_to)

    if k is None:
        receivers_list = tree.query_radius(positions_from, r=radius)
        num_nodes = len(positions_from)
        indices_from = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
        indices_to = np.concatenate(receivers_list, axis=0)
    else:

        dists, receivers_list = tree.query(positions_from, k=k)
        indices_from = []
        indices_to = []

        for i in range(dists.shape[0]):
            ind_from = i
            for j in range(dists.shape[1]):
                if dists[i, j] <= radius:
                    ind_to = receivers_list[i, j]
                    indices_from.append(ind_from)
                    indices_to.append(ind_to)

        indices_from = np.array(indices_from).astype(np.int64)
        indices_to = np.array(indices_to).astype(np.int64)
    return indices_from, indices_to


def compute_connectivity_pt(positions_from, positions_to, radius, k=None):
    device = positions_from.device
    positions_from = positions_from.detach().cpu().numpy()
    positions_to = positions_to.detach().cpu().numpy()
    indices_from, indices_to = compute_connectivity_np(positions_from, positions_to, radius, k=k)

    indices_from = torch.LongTensor(indices_from).to(device)
    indices_to = torch.LongTensor(indices_to).to(device)

    return indices_from, indices_to
