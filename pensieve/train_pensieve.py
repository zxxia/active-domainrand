import argparse

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
    # parser.add_argument('--RANDOM_SEED', type=int, default=42, help='')
    parser.add_argument('--total-epoch', type=int, default=50000,
                        help='Total training epoch.')

    # data related paths
    parser.add_argument("--video-size-file-dir", type=str, required=True,
                        help='Dir to video size files')
    parser.add_argument("--train-env-config", type=str, required=True,
                        help='Path to training environment configuration.')
    parser.add_argument("--val-env-config", type=str, required=True,
                        help='Path to training environment configuration.')
    parser.add_argument("--test-env-config", type=str, required=True,
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


def main():
    args = parse_args()
    pensieve_abr = Pensieve(args.num_agents, args.summary_dir,
                            model_save_interval=args.model_save_interval,
                            batch_size=args.batch_size)

    # prepare train dataset
    traces_time, traces_bw, traces_names = load_traces(args.train_trace_dir)
    train_envs = []
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(traces_time, traces_bw, traces_names)):
        net_env = Environment(args.video_size_file_dir, args.train_env_config,
                              trace_idx, trace_time=trace_time,
                              trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=False,
                              trace_video_same_duration_flag=True)
        train_envs.append(net_env)

    # prepare train dataset
    traces_time, traces_bw, traces_names = load_traces(args.val_trace_dir)
    val_envs = []
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(traces_time, traces_bw, traces_names)):
        net_env = Environment(args.video_size_file_dir, args.val_env_config,
                              trace_idx, trace_time=trace_time,
                              trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        val_envs.append(net_env)

    # prepare test dataset
    test_envs = []
    traces_time, traces_bw, traces_names = load_traces(args.test_trace_dir)
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(traces_time, traces_bw, traces_names)):
        net_env = Environment(args.video_size_file_dir, args.test_env_config,
                              trace_idx, trace_time=trace_time,
                              trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        test_envs.append(net_env)

    # test training
    pensieve_abr.train(train_envs, val_envs=val_envs, test_envs=test_envs,
                       iters=args.total_epoch)


if __name__ == "__main__":
    main()