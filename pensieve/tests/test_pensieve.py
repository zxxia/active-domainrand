import time
import csv
import os

from pensieve.agent_policy import Pensieve
from pensieve.environment import Environment
from pensieve.utils import load_traces

VIDEO_SIZE_FILE_DIR = '/data3/zxxia/pensieve/data/video_sizes'
CONFIG_FILE = '/data3/zxxia/active-domainrand/pensieve/config/default.json'
SEED = 42
SUMMARY_DIR = '/data3/zxxia/active-domainrand/pensieve/tests/pensieve_log'
TEST_TRACE_DIR = '/data3/zxxia/pensieve/data/test'
VAL_TRACE_DIR = '/data3/zxxia/pensieve/data/val'
TRAIN_TRACE_DIR = '/data3/zxxia/pensieve/data/train'

ACTOR_PATH = '/data3/zxxia/pensieve-pytorch/results_fix_pred/actor.pt'
CRITIC_PATH = '/data3/zxxia/pensieve-pytorch/results_fix_pred/actor.pt'


def main():
    pensieve_abr = Pensieve(16, SUMMARY_DIR, actor_path=ACTOR_PATH,
                            model_save_interval=10, batch_size=100)

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        TRAIN_TRACE_DIR)
    net_envs = []
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(all_cooked_time, all_cooked_bw, all_file_names)):
        net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, trace_idx,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=False,
                              trace_video_same_duration_flag=True)
        net_envs.append(net_env)

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(VAL_TRACE_DIR)
    val_envs = []
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(all_cooked_time, all_cooked_bw, all_file_names)):
        net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, trace_idx,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        val_envs.append(net_env)

    test_envs = []
    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        TEST_TRACE_DIR)
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(all_cooked_time, all_cooked_bw, all_file_names)):
        net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, trace_idx,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        test_envs.append(net_env)

    # test training
    pensieve_abr.train(net_envs, val_envs=val_envs, test_envs=test_envs, iters=1e4)

    # test evaluate and evalute_env
    t_start = time.time()
    results = pensieve_abr.evaluate_envs(test_envs, os.path.join(SUMMARY_DIR, 'test_log'))
    print('multiproc', time.time() - t_start)
    t_start = time.time()
    results = pensieve_abr.evaluate_envs(test_envs)
    print('singleproc', time.time() - t_start)


if __name__ == "__main__":
    main()
