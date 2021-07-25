from collections import defaultdict

from nasbench import api as api101

from utils import edit_one_edge, edit_one_node

DEFAULT_DATA_PATH = "../data/nasbench_only108.tfrecord"


def matrix_unique_set(dataset):
    matrix_set = set()
    for _, v in enumerate(dataset.fixed_statistics.values()):
        matrix = v['module_adjacency']
        matrix_tupled = tuple([tuple(row) for row in matrix])
        matrix_set.add(matrix_tupled)
    return matrix_set


def ops_unique_set(dataset):
    ops_set = set()
    for _, v in enumerate(dataset.fixed_statistics.values()):
        ops = v["module_operations"]
        ops_tupled = tuple(ops)
        ops_set.add(ops_tupled)
    return ops_set


def groupby_len(data_list):
    dict_by_len = defaultdict(list)
    for data in data_list:
        dict_by_len[len(data)].append(data)
    return dict_by_len


def get_hash_from_arch(arch_list, is_valid, hash_spec):
    hash_list = []
    for arch in arch_list:
        model_spec = api101.ModelSpec(arch[0], arch[1])
        if is_valid(model_spec):
            module_hash = hash_spec(model_spec)
            hash_list.append(module_hash)
    return hash_list


dataset = api101.NASBench(DEFAULT_DATA_PATH)

ops_set = ops_unique_set(dataset)
ops_dict_by_len = groupby_len(ops_set)

matrix_set = matrix_unique_set(dataset)
for matrix in matrix_set:
    for ops in ops_dict_by_len[len(matrix)]:
        edge_edit_hashes = get_hash_from_arch(edit_one_edge(matrix, ops),
                                              is_valid=dataset.is_valid,
                                              hash_spec=dataset._hash_spec
                                              )
        node_edit_hashes = get_hash_from_arch(edit_one_node(matrix, ops),
                                              is_valid=dataset.is_valid,
                                              hash_spec=dataset._hash_spec
                                              )

        # list of edit one edge
        # list of edit one node
