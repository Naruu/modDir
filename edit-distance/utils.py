import csv
import glob
from copy import deepcopy

import networkx as nx

OP_CHOICES = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]


def partition(lst, num_partition):
    for i in range(0, len(lst), num_partition):
        yield lst[i: i + num_partition]


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


def unique_edgelist(input_dir, output_filename):
    c = 0
    hash_pairs = set()
    for filename in glob.glob(input_dir + "/*.csv"):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                hash_pairs.add(tuple(sorted(row)))
                c += 1
    print(f"number of hash pairs. raw: {c}, unique: {len(hash_pairs)}")

    output_dir = "/".join(output_filename.split("/")[:-1])
    os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(hash_pairs)
