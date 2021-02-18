import argparse
import json
import logging
import os

import numpy as np

from pensieve.agent_policy import Pensieve
from pensieve.environment import Environment
from pensieve.utils import load_traces


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Train Pensieve")
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description of the experiment.')
    # Training related settings
    parser.add_argument('--num-agents', type=int, default=16,
                        help='Num of worker agents. Defaults to 16.')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Take as a train batch.')
    parser.add_argument('--model-save-interval', type=int, default=100,
                        help='Save model every n training iterations.')
    parser.add_argument('--randomization-interval', type=int, default=1,
                        help='How frequent UDR occurs. Default:')
    # parser.add_argument('--RANDOM_SEED', type=int, default=42, help='')
    parser.add_argument('--total-epoch', type=int, default=50000,
                        help='Total training epoch.')
    parser.add_argument('--randomization', type=str, default='',
                        choices=['', 'udr', 'adr', 'even_udr'],
                        help='Mode of domain randomization. \'\' means no '
                        'randomization. \'udr\' means uniform domain '
                        'randomization. Every trainng epoch uniformly sampling'
                        ' over a large range. \'even_udr\' means it splits the'
                        ' large range into num-agents small spaces. Each agent'
                        ' uniformly samples a small range. \'adr\' means '
                        'active domain randomization.')

    # data related paths
    parser.add_argument("--video-size-file-dir", type=str, required=True,
                        help='Dir to video size files')
    parser.add_argument("--train-env-config", type=str, required=True,
                        help='Path to training environment configuration.')
    parser.add_argument("--val-env-config", type=str, default=None,
                        help='Path to training environment configuration.')
    parser.add_argument("--test-env-config", type=str, default=None,
                        help='Path to training environment configuration.')
    parser.add_argument("--train-trace-dir", type=str, default=None,
                        help='Dir to all train traces. When None, then use'
                        'the simulator generated traces.')
    parser.add_argument("--val-trace-dir", type=str, default=None,
                        help='Dir to all val traces. When None, then use the'
                        ' simulator generated traces.')
    parser.add_argument("--test-trace-dir", type=str, default=None,
                        help='Dir to all test traces. When None, then use '
                        'the simulator generated traces.')

    # model related paths
    parser.add_argument("--summary-dir", type=str, required=True,
                        help='Folder to save all training results.')
    parser.add_argument("--nn-model", type=str, default=None,
                        help='model path')
    # parser.add_argument("--env-random-start", action="store_true",
    #                     help='environment will randomly start a new trace'
    #                     'in training stage if environment is not fixed if '
    #                     'specified.')

    return parser.parse_args()


def log_args(args):
    """Write arguments to log. Assumes args.results_dir exists."""
    os.makedirs(args.summary_dir, exist_ok=True)
    log_file = os.path.join(args.summary_dir, 'args')
    config_logging = logging.getLogger("args")
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    config_logging.setLevel(logging.INFO)
    config_logging.addHandler(file_handler)
    config_logging.addHandler(stream_handler)
    for arg in vars(args):
        config_logging.info(arg + '\t' + str(getattr(args, arg)))


def main():
    args = parse_args()
    log_args(args)
    pensieve_abr = Pensieve(args.num_agents, args.summary_dir,
                            model_save_interval=args.model_save_interval,
                            batch_size=args.batch_size,
                            randomization=args.randomization,
                            randomization_interval=args.randomization_interval)

    # prepare train dataset
    if args.train_trace_dir is not None:
        traces_time, traces_bw, traces_names = load_traces(
            args.train_trace_dir)
        train_envs = []
        for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
                zip(traces_time, traces_bw, traces_names)):
            net_env = Environment(args.video_size_file_dir,
                                  args.train_env_config, trace_idx,
                                  trace_time=trace_time, trace_bw=trace_bw,
                                  trace_file_name=trace_filename, fixed=False,
                                  trace_video_same_duration_flag=True)
            train_envs.append(net_env)
    else:
        train_envs = [Environment(args.video_size_file_dir,
                                  args.train_env_config, i, fixed=False,
                                  trace_video_same_duration_flag=True)
                                  for i in range(100)]

    # prepare train dataset
    val_envs = None
    if args.val_trace_dir is not None:
        traces_time, traces_bw, traces_names = load_traces(args.val_trace_dir)
        val_envs = []
        for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
                zip(traces_time, traces_bw, traces_names)):
            net_env = Environment(args.video_size_file_dir, args.
                                  val_env_config,
                                  trace_idx, trace_time=trace_time,
                                  trace_bw=trace_bw,
                                  trace_file_name=trace_filename, fixed=True,
                                  trace_video_same_duration_flag=True)
            val_envs.append(net_env)

        # with open(args.val_env_config, mode='r') as f:
        #     config = json.load(f)
        # dimensions = []
        # for dimension in config['dimensions']:
        #     if dimension['name'] in ["buffer_threshold",  "link_rtt",
        #                              "drain_buffer_sleep_time",
        #                              "packet_payload_portion"]:
        #         dimensions.append(
        #             np.linspace(dimension['min'], dimension['max'], num=3))
        #     # dimensions[dimension['name']] = Dimension(
        #     #     default_value=dimension['default'],
        #     #     seed=self.seed,
        #     #     min_value=dimension['min'],
        #     #     max_value=dimension['max'],
        #     #     name=dimension['name'],
        #     #     unit=dimension['unit'])

    # prepare test dataset
    test_envs = None
    if args.test_trace_dir is not None:
        test_envs = []
        traces_time, traces_bw, traces_names = load_traces(args.test_trace_dir)
        for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
                zip(traces_time, traces_bw, traces_names)):
            net_env = Environment(args.video_size_file_dir,
                                  args.test_env_config, trace_idx,
                                  trace_time=trace_time, trace_bw=trace_bw,
                                  trace_file_name=trace_filename, fixed=True,
                                  trace_video_same_duration_flag=False)
            test_envs.append(net_env)

    # test training
    pensieve_abr.train(train_envs, val_envs=val_envs, test_envs=test_envs,
                       iters=args.total_epoch)


if __name__ == "__main__":
    main()
