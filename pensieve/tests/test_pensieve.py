from pensieve.env import Environment
from pensieve.pensieve import Pensieve
from pensieve.utils import linear_reward, load_traces

VIDEO_SIZE_FILE_DIR = '/data3/zxxia/pensieve/data/video_sizes'
CONFIG_FILE = '/data3/zxxia/active-domainrand/pensieve/config/default.json'
SEED = 42
SUMMARY_DIR = '/data3/zxxia/active-domainrand/pensieve/tests/pensieve_log'
TEST_TRACE_DIR = '/data3/zxxia/pensieve/data/test'

pensieve_abr = Pensieve(2, SUMMARY_DIR, model_save_interval=1)

all_cooked_time, all_cooked_bw, all_file_names = load_traces(
    TEST_TRACE_DIR)
net_envs = []
for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
        zip(all_cooked_time, all_cooked_bw, all_file_names)):
    net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, SEED,
                          trace_time=trace_time, trace_bw=trace_bw,
                          trace_file_name=trace_filename, fixed=False,
                          trace_video_same_duration_flag=True)
    net_envs.append(net_env)

pensieve_abr.train(net_envs, 15)
