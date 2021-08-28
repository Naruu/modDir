import argparse
import csv
import datetime
import os
import glob
import logging
import time
from multiprocessing import Pool

from subprocess import Popen, PIPE

from nasbench import api as api101

from utils import partition


def generate_args(binary, *params):
    arguments = [binary]
    arguments.extend(list(params))
    return arguments


def execute_binary(args):
    process = Popen(' '.join(args), shell=True, stdout=PIPE,
                    stderr=PIPE, encoding='utf8')
    (std_output, std_error) = process.communicate()
    process.wait()
    rc = process.returncode

    return rc, std_output, std_error


def parse_embedding_num(std_output):
    embedding_num = 0
    for line in std_output.split('\n'):
        if '#Embeddings' in line:
            embedding_num = int(line.split(':')[1].strip())
    return embedding_num


def get_subgraph_relation(binary_path, data_graph_path_list, query_dir_path, output_path, dataset):
    query_graph_path_list = glob.glob(f'{query_dir_path}/*.graph')

    for d_idx, data_graph_path in enumerate(data_graph_path_list):
        logging.info(f"data_graph idx: {d_idx}th iteration")
        data_graph_hash = os.path.splitext(
            os.path.basename(data_graph_path))[0]
        data_graph_len = len(dataset.get_metrics_from_hash(
            data_graph_hash)[0]['module_operations'])

        for q_idx, query_graph_path in enumerate(query_graph_path_list):
            if q_idx % 100 == 0:
                logging.info(f"{d_idx}, {q_idx}th iteration")
            query_graph_hash = os.path.splitext(
                os.path.basename(query_graph_path))[0]
            query_graph_len = len(dataset.get_metrics_from_hash(
                query_graph_hash)[0]['module_operations'])

            if query_graph_len > data_graph_len:
                continue

            execution_args = generate_args(binary_path, '-d', data_graph_path, '-q', query_graph_path, '-filter', 'LDF',
                                           '-order', 'QSI', '-engine', 'QSI', '-num', '1')
            (rc, std_output, std_error) = execute_binary(execution_args)

            if rc != 0:
                logging.error(
                    f'Error: data hash: {data_graph_hash}, query hash: {query_graph_hash}')
                logging.error(std_error)

            embedding_num = parse_embedding_num(std_output)
            if embedding_num > 0:
                with open(output_path, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([query_graph_hash, data_graph_hash])


def single_process_job(data_graph_list, dataset, output_dir, query_path, binary_path):
    log_dir = f"{output_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)

    pid = os.getpid()
    output_path = f"{output_dir}/{pid}.csv"
    log_path = f"{log_dir}/{pid}.log"

    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(processName)s %(message)s")
    logging.info(f"number of graphs: {len(data_graph_list)}")

    return get_subgraph_relation(binary_path, data_graph_list, query_path, output_path, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nd', '--nasbench_data_path')
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-q', '--query_path')
    parser.add_argument('-b', '--binary_path')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('-p', '--num_processes', type=int, required=True,
                        help=f'number of processes')

    args = parser.parse_args()
    cur_dt = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    output_dir = args.output_dir + "/" + cur_dt

    dataset = api101.NASBench(args.nasbench_data_path)
    os.makedirs(output_dir, exist_ok=True)

    data_graph_path_list = glob.glob(f'{args.data_path}/*.graph')
    data_graph_partitions = partition(data_graph_path_list, args.num_processes)
    n_graphs = len(data_graph_path_list)
    print(
        f"number of graphs: {n_graphs}, approximately per process: {n_graphs / args.num_processes}")

    p_args = [dataset, output_dir, args.query_path, args.binary_path]
    p_args_list = [[p_graphs] + p_args for [p_graphs] in data_graph_partitions]

    start = time.time()
    with Pool(args.num_processes) as p:
        p.starmap(single_process_job, p_args_list)
    end = time.time()
    print(f"time spent: {end - start}")
