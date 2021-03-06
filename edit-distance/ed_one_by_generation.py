import time
import argparse
import csv
from collections import defaultdict
import datetime
from multiprocessing import Pool
import logging
import os
import random

import numpy as np

from nasbench import api as api101

from utils import edit_one_edge, edit_one_node

# DEFAULT_DATA_PATH = ".data/nasbench_only108.tfrecord"
DEFAULT_DATA_PATH = "../data/500.tfrecord"

SEED_ARCH_COUNT = 20000
OUTPUT_DIR = "outputs"

random.seed(0)


def partition(lst, num_partition):
    for i in range(0, len(lst), num_partition):
        yield lst[i: i + num_partition]


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


def archs_to_hashes(arch_list, dataset):
    hash_list = []
    for matrix, ops in arch_list:
        model_spec = api101.ModelSpec(matrix, ops)
        if model_spec.ops == ops \
                and (model_spec.matrix == matrix).all() \
                and dataset.is_valid(model_spec):
            module_hash = dataset._hash_spec(model_spec)
            hash_list.append(module_hash)
    return hash_list


def single_arch_job(matrix, ops, dataset, output_path):
    edge_edited = edit_one_edge(matrix, ops)
    node_edited = edit_one_node(matrix, ops)

    origin_hash = archs_to_hashes([(matrix, ops)], dataset)[0]
    edited_hashes = archs_to_hashes(edge_edited + node_edited, dataset)
    rows = [(origin_hash, edited_hash) for edited_hash in edited_hashes]

    with open(output_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def single_process_job(p_arg):
    dataset, output_dir, log_dir, ops_dict_by_len, matrix_set = p_arg

    output_path = f"{output_dir}/{os.getpid()}.csv"
    log_path = f"{log_dir}/{os.getpid()}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(processName)s %(message)s")
    logging.info(f"number of matrix: {len(matrix_set)}")

    for i, matrix_tupled in enumerate(matrix_set):
        if i % 30 == 0:
            logging.info(f"{i}th iteration.")
        matrix = np.array(matrix_tupled)
        for ops_tupled in ops_dict_by_len[len(matrix[0])]:
            ops = list(ops_tupled)
            single_arch_job(matrix, ops, dataset, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=False, default=DEFAULT_DATA_PATH,
                        help=f'path to nasbench101 dataset. default: {DEFAULT_DATA_PATH}')
    parser.add_argument('-n', '--seed_arch_count', type=int, required=False,
                        help=f'number_of_seed_architectures')
    parser.add_argument('-o', '--output_dir', required=False, default=OUTPUT_DIR,
                        help=f'path to output directory, default "{OUTPUT_DIR}"')
    parser.add_argument('-p', '--num_processes', type=int, required=True,
                        help=f'number of processes')

    args = parser.parse_args()
    cur_dt = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    output_dir = args.output_dir + "/" + cur_dt
    log_dir = "logs/" + cur_dt

    dataset = api101.NASBench(args.data_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ops_set = ops_unique_set(dataset)
    ops_dict_by_len = groupby_len(ops_set)

    matrix_list = list(matrix_unique_set(dataset))
    matrix_partitions = partition(matrix_list, args.num_processes)
    len_matrix = len(matrix_list)
    print(
        f"number of matrices: {len_matrix}, approximately per process: {len_matrix / args.num_processes}")

    p_args = [dataset, output_dir, log_dir, ops_dict_by_len]
    p_args_list = [p_args + [p_matrix] for p_matrix in matrix_partitions]

    start = time.time()
    with Pool(args.num_processes) as p:
        p.map(single_process_job, p_args_list)
    end = time.time()
    print(f"time spent: {end - start}")
