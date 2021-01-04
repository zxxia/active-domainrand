import sys
sys.path.append("/Users/lesley/ADR-Pensieve")

import argparse
import csv
import itertools
import os
import time

import numpy as np

from pensieve.agent_policy import Pensieve, RobustMPC
from pensieve.environment import Environment
from pensieve.utils import load_traces


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Evaluate ABR under different config.")
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description of the experiment.')
    # Training related settings
    # parser.add_argument('--RANDOM_SEED', type=int, default=42, help='')
    parser.add_argument('--total-epoch', type=int, default=50000,
                        help='Total training epoch.')

    # data related paths
    parser.add_argument("--video-size-file-dir", type=str, required=True,
                        help='Dir to video size files')
    parser.add_argument("--test-env-config", type=str, required=True,
                        help='Path to training environment configuration.')
    parser.add_argument("--test-trace-dir", type=str, default=None,
                        help='Dir to all test traces. When None, then use '
                        'the simulator generated traces.')
    parser.add_argument("--dataset-name", type=str, default=None,
                        help='Name of testing dataset.')

    # model related paths
    parser.add_argument("--summary-dir", type=str, required=True,
                        help='Folder to save all training results.')
    parser.add_argument("--actor-path", type=str, default=None,
                        help='model path')

    # abr agent policy related
    parser.add_argument("--abr", type=str, required=True, default='pensieve',
                        choices=['pensieve', 'mpc'],
                        help='supported ABR algorithm.')
    # parser.add_argument("--env-random-start", action="store_true",
    #                     help='environment will randomly start a new trace'
    #                     'in training stage if environment is not fixed if '
    #                     'specified.')

    return parser.parse_args()


def main():
    args = parse_args()

    # prepare test dataset
    link_rtt_list = [10, 100, 1000, 10000]
    buf_thresh_list = [20, 30, 60, 160]
    drain_buffer_time_list = [100, 400, 800, 1000]
    packet_payload_portion_list = [0.5, 0.7, 0.9, 1.0]
    nb_config_combos = len(buf_thresh_list) * len(link_rtt_list) * \
        len(drain_buffer_time_list) * len(packet_payload_portion_list)
    if args.abr == 'pensieve':
        abr = Pensieve(1, args.summary_dir, actor_path=args.actor_path)
        log_filename = '{}_test_{}_results.csv'.format(
            os.path.splitext(os.path.basename(args.actor_path))[0],
            args.dataset_name)
    elif args.abr == 'mpc':
        abr = RobustMPC()
        log_filename = 'mpc_test_{}_results.csv'.format(args.dataset_name)
    else:
        raise NotImplementedError
    csv_writer = csv.writer(open(os.path.join(args.summary_dir, log_filename),
                                 'w', 1), lineterminator='\n')
    csv_writer.writerow(['buffer_threshold', 'link_rtt',
                         'drain_buffer_sleep_time', 'packet_payload_portion',
                         'avg_chunk_reward'])

    test_envs = []
    traces_time, traces_bw, traces_names = load_traces(args.test_trace_dir)
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(traces_time, traces_bw, traces_names)):
        net_env = Environment(args.video_size_file_dir,
                              args.test_env_config, trace_idx,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        test_envs.append(net_env)
    for i, (buf_thresh, link_rtt, drain_buffer_time, pkt_payload_portion) in \
        enumerate(itertools.product(buf_thresh_list, link_rtt_list,
                                    drain_buffer_time_list,
                                    packet_payload_portion_list)):
        for net_env in test_envs:
            net_env.reset()
            net_env.randomize(
                {'buffer_threshold': round(buf_thresh, 6),
                 'link_rtt': round(link_rtt, 6),
                 'drain_buffer_sleep_time': round(drain_buffer_time, 6),
                 'packet_payload_portion': round(pkt_payload_portion, 6)})

        # test training
        if args.abr == 'pensieve':
            t_start = time.time()
            results = abr.evaluate_envs(test_envs)
            vid_rewards = [np.array(vid_results)[1:, -1]
                           for vid_results in results]
            avg_chunk_reward = np.mean(np.concatenate(vid_rewards))
            print('pensieve: {:.5f}s'.format(time.time() - t_start))
        elif args.abr == 'mpc':
            t_start = time.time()
            results = abr.evaluate_envs(test_envs)
            vid_rewards = [np.array(vid_results)[1:, -2]
                           for vid_results in results]
            avg_chunk_reward = np.mean(np.concatenate(vid_rewards))
            print('mpc: {:.5f}s'.format(time.time() - t_start))
        else:
            raise NotImplementedError
        csv_writer.writerow([buf_thresh, link_rtt, drain_buffer_time,
                             pkt_payload_portion, avg_chunk_reward])
        print("{}/{}".format(i, nb_config_combos))


if __name__ == "__main__":
    main()
