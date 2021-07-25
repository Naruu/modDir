import argparse
import csv
import datetime
from multiprocessing import Pool
import logging
import os
import random

import networkx as nx

from nasbench import api as api101

from utils import get_edit_distance

# DEFAULT_DATA_PATH = ".data/nasbench_only108.tfrecord"
DEFAULT_DATA_PATH = "../data/500.tfrecord"

SEED_ARCH_COUNT = 20000
MAX_EDIT_DISTANCE = 1
OUTPUT_DIR = "outputs"

random.seed(0)

def partition(N, num_partition):
    partitions = [0] * num_partition
    p_sums = [0] * num_partition
    count_per_partition = (N-1) * N / 2 / num_partition
    print(f"count_per_partition: {count_per_partition}")

    p_idx = 0
    for row_idx in range(N):
        col_count = N - row_idx - 1
        p_sums[p_idx] += col_count

        if p_sums[p_idx] > count_per_partition:
            p_idx += 1
            partitions[p_idx] = row_idx + 1
    p_range = [(partitions[i], partitions[i+1]) for i in range(num_partition-1)]
    p_range.append((partitions[-1], N-1))
    print(f" p_sums: {p_sums}, partitions: {partitions}, p_range: {p_range}")
    return p_range


def edit_distance_job(arg):
    dataset, seed_hashes, output_dir, log_dir, max_edit_distance, idxs = arg

    log_path = f"{log_dir}/{os.getpid()}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(processName)s %(message)s")
    output_path = f"{output_dir}/{os.getpid()}.csv"

    first_idx = idxs[0]
    from_hashes = seed_hashes[idxs[0]: idxs[1]]

    for from_idx, from_hash in enumerate(from_hashes):
        if from_idx % 100 == 0:
            logging.info(f"pid:{os.getpid()}, current: {from_idx}, between: ({idxs[0]}, {idxs[1]}) {datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}")

        from_model = dataset.fixed_statistics[from_hash]
        seed_matrix, seed_ops = from_model['module_adjacency'], from_model['module_operations']

        global_idx = first_idx + from_idx
        to_hashes = seed_hashes[global_idx:]
        for _, to_hash in enumerate(to_hashes):
            print(_)
            print(to_hash)
            to_model = dataset.fixed_statistics[to_hash]
            matrix, ops = to_model['module_adjacency'], to_model['module_operations']

            true_distance = get_edit_distance(seed_matrix, seed_ops, matrix, ops, max_edit_distance)

            if true_distance > 0:
                with open(output_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([from_hash, to_hash, true_distance])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys_path', type=str, required=False,
                        help=f'path to nasbench101 hash keys')
    parser.add_argument('-d', '--data_path', type=str, required=False, default=DEFAULT_DATA_PATH,
                        help=f'path to nasbench101 dataset. default: {DEFAULT_DATA_PATH}')
    parser.add_argument('-n', '--seed_arch_count', type=int, required=False,
                         help=f'number_of_seed_architectures')
    parser.add_argument('-ed', '--max_edit_distance', type=int, default=MAX_EDIT_DISTANCE,
                        help=f'max edit distance to look up from dataset. default: {MAX_EDIT_DISTANCE}')
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

    if args.keys_path:
        with open(args.keys_path, 'r') as f:
            hash_keys = [key.strip() for key in f.readlines()]
    else:
        hash_keys = list(dataset.hash_iterator())
        with open("../data/hash_keys.txt", 'w') as f:
            for key in hash_keys:
                f.write(key + "\n")

    if args.seed_arch_count:
        seed_hashes = random.sample(hash_keys, args.seed_arch_count)
    else:
        seed_hashes = hash_keys

    # seed_hashes = list(dataset.hash_iterator())
    print(f"number of hashe: {len(seed_hashes)}")

    p_args = [dataset, seed_hashes, output_dir, log_dir, args.max_edit_distance]
    idxs = partition(len(seed_hashes), args.num_processes)
    p_args_list = [p_args + [idxs[i]] for i in range(len(idxs))]

    with Pool(args.num_processes) as p:
        p.map(edit_distance_job, p_args_list)
