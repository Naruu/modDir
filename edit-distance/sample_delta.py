import time
import argparse
import csv
import datetime
from multiprocessing import Pool
import logging
import os
import time
import numpy as np
import networkx as nx

from utils import partition


def single_process_job(node_tuples, log_dir):
    log_path = f"{log_dir}/{os.getpid()}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(processName)s %(message)s")
    logging.info(f"Process started")

    hyps = []
    for i, node_tuple in enumerate(node_tuples):
        if i % 50 == 0:
            logging.info(f"{i}th iteration")

        s = []
        try:
            d01 = nx.shortest_path_length(
                G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(
                G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(
                G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(
                G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(
                G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(
                G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue

    if hyps:
        return max(hyps)
    else:
        return -1


def hyperbolicity_sample(G, log_dir, seed=0, num_samples=50000, num_processes=1):
    np.random.seed(seed)
    node_tuples = []

    for _ in range(num_samples):
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        node_tuples.append(node_tuple)

    nodes_partitioned = partition(node_tuples, num_processes)
    p_args_list = [(p_nodes, log_dir) for p_nodes in nodes_partitioned]
    for p_arg in p_args_list:
        print(p_arg)

    hyps = []
    with Pool(num_processes) as p:
        hyps = p.starmap(single_process_job, p_args_list)

    if hyps:
        return max(hyps)
    else:
        return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help=f'path to edgelist csv file.')
    parser.add_argument('-n', '--num_samples', type=int, required=True)
    parser.add_argument('-s', '--seed', type=int, required=False, default=0)
    parser.add_argument('-o', '--output_dir', required=False, default="outputs-hyps",
                        help=f'path to output dir')
    parser.add_argument('-p', '--num_processes', type=int, required=True,
                        help=f'number of processes')

    args = parser.parse_args()
    cur_dt = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    output_dir = args.output_dir + "/" + cur_dt
    log_dir = "logs/" + cur_dt

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    read_start = time.time()
    with open(args.data_path) as csvfile:
        reader = csv.reader(csvfile)
        G = nx.from_edgelist(reader)

    print(
        f"Finished reading data from {args.data_path}. {time.time() - read_start} took secs")

    start_time = time.time()
    delta = hyperbolicity_sample(G, log_dir, args.seed,
                                 args.num_samples, args.num_processes)
    end_time = time.time()

    with open(output_dir + "-delta.txt", "w") as f:
        f.write(
            f"data: {args.data_path}, seed: {args.seed}, num_samples: {args.num_samples}, delta: {delta}\n")
        f.write(
            f"time elapsed: {end_time - start_time:.3f} sec with {args.num_processes} processes\n")
