from copy import deepcopy

import networkx as nx
import numpy as np

OP_CHOICES = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]


def to_nx_graph(matrix, ops):
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    for idx, op in enumerate(ops):
        G.add_node(idx, operation=op)
    return G


def node_match(node1, node2):
    return node1["operation"] == node2["operation"]


def edge_match(edge1, edge2):
    return edge1 == edge2


def get_edit_distance(matrix1, ops1, matrix2, ops2, upper_bound):
    G1 = to_nx_graph(matrix1, ops1)
    G2 = to_nx_graph(matrix2, ops2)

    return nx.graph_edit_distance(G1, G2, node_match=node_match, edge_match=edge_match, upper_bound=upper_bound) or -1


def matrix_unique_set(dataset):
    matrix_set = set()
    for _, v in enumerate(dataset.fixed_statistics.values()):
        matrix = v['module_adjacency']
        matrix_tupled = tuple([tuple(row) for row in matrix])
        matrix_set.add(matrix_tupled)
    return matrix_set


def partition(lst, num_partition):
    for i in range(0, len(lst), num_partition):
        yield lst[i: i + num_partition]


def edit_one_edge(original_matrix, original_ops):
    edited = []
    ops = original_ops

    len_matrix = len(original_matrix[0])
    for i in range(len_matrix - 1):
        for j in range(i + 1, len_matrix):
            matrix = deepcopy(original_matrix)
            matrix[i][j] = 1 - matrix[i][j]
            edited.append((matrix, ops))
    return edited


def delete_one_edge(original_matrix, original_ops):
    edited = []
    ops = original_ops

    len_matrix = len(original_matrix[0])
    for i in range(len_matrix - 1):
        for j in range(i + 1, len_matrix):
            if original_matrix[i][j]:
                matrix = deepcopy(original_matrix)
                matrix[i][j] = 1 - matrix[i][j]
                edited.append((matrix, ops))
    return edited


def delete_two_edges(original_matrix, original_ops):
    edited = []
    ops = original_ops

    row, col = np.where(original_matrix > 0)
    indices = list(zip(row, col))
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            matrix = deepcopy(original_matrix)
            matrix[indices[i]] = 0
            matrix[indices[j]] = 0
            edited.append((matrix, ops))
    return edited


def delete_one_node_one_edge(original_matrix, ops):
    edited = []

    len_matrix = len(original_matrix[0])
    v_sum = np.sum(original_matrix, axis=0)
    h_sum = np.sum(original_matrix, axis=1)
    for i in range(1, len_matrix - 1):
        if (v_sum[i] + h_sum[i]) == 1:
            matrix = original_matrix
            matrix = np.delete(matrix, i, 0)
            matrix = np.delete(matrix, i, 1)
            ops = np.delete(ops, i, 0)
            edited.append((matrix, ops))
    return edited


def edit_one_node(original_matrix, original_ops):
    edited = []
    matrix = original_matrix

    for idx in range(1, len(original_ops)-1):
        for choice in OP_CHOICES:
            if choice == original_ops[idx]:
                continue
            ops = deepcopy(original_ops)
            ops[idx] = choice
            edited.append((matrix, ops))
    return edited
