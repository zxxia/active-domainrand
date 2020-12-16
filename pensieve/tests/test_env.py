import argparse
import itertools
import multiprocessing as mp
import os
import time

import numpy as np
from pensieve.constants import (A_DIM, B_IN_MB, BUFFER_NORM_FACTOR,
                                DEFAULT_QUALITY, M_IN_K, REBUF_PENALTY, S_INFO,
                                S_LEN, SMOOTH_PENALTY, TOTAL_VIDEO_CHUNK,
                                VIDEO_BIT_RATE)
from numba import jit
from pensieve.utils import linear_reward, load_traces

from pensieve.env import Environment

MPC_FUTURE_CHUNK_COUNT = 5
VIDEO_BIT_RATE = np.array(VIDEO_BIT_RATE)  # Kbps
VIDEO_SIZE_FILE_DIR = '/data3/zxxia/pensieve/data/video_sizes'
CONFIG_FILE = '/data3/zxxia/active-domainrand/pensieve/config/default.json'
SEED = 42
SUMMARY_DIR = '/data3/zxxia/active-domainrand/pensieve/tests/mpc_logs'
TEST_TRACE_DIR = '/data3/zxxia/pensieve/data/test'


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Pensieve testing script.")
#     parser.add_argument("--test_trace_dir", type=str,
#                         required=True, help='dir to all test traces.')
#     parser.add_argument("--summary_dir", type=str,
#                         required=True, help='output path.')
#     parser.add_argument("--random_seed", type=int, default=42)
#     parser.add_argument("--video-size-file-dir", type=str, required=True,
#                         help="dir to all video size files.")
#     parser.add_argument("--env-random-start", action="store_true",
#                         help='environment will randomly start a new trace'
#                         'in training stage if environment is not fixed if '
#                         'specified.')
#     parser.add_argument("--buffer-thresh", type=float, default=60,
#                         help='buffer threshold(sec)')
#     parser.add_argument("--link-rtt", type=float, default=80,
#                         help='link RTT(millisec)')
#     parser.add_argument("--drain-buffer-sleep-time", type=float, default=500,
#                         help='drain buffer sleep time(millisec)')
#     parser.add_argument("--packet-payload-portion", type=float, default=0.95,
#                         help='drain buffer sleep time(millisec)')
#
#     return parser.parse_args()


@jit(nopython=True)
def calculate_rebuffer(future_chunk_length, buffer_size, bit_rate, last_index,
                       future_bandwidth, video_size, chunk_combo_options):
    max_reward = -100000000
    best_combo = ()
    start_buffer = buffer_size

    for full_combo in chunk_combo_options:
        # print(type(future_chunk_length))
        combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with
        # start_buffer and subtract each download time and add 2 seconds in
        # that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int(bit_rate)
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            # e.g., if last chunk is 3, then first iter is 3+0+1=4
            index = last_index + position + 1
            # this is MB/MB/s --> seconds
            download_time = \
                video_size[chunk_quality, index % TOTAL_VIDEO_CHUNK] / \
                B_IN_MB / future_bandwidth
            if (curr_buffer < download_time):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += 4
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            smoothness_diffs += SMOOTH_PENALTY * abs(
                VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in
        # Mbits/s

        reward = (bitrate_sum / M_IN_K) - \
            (REBUF_PENALTY * curr_rebuffer_time) - \
            (smoothness_diffs / M_IN_K)

        if reward >= max_reward:
            # if (best_combo != ()) and best_combo[0] < combo[0]:
            #     best_combo = combo
            # else:
            best_combo = combo

            max_reward = reward
            # send data to html side (first chunk of best combo)
            # no combo had reward better than -1000000 (ERROR) so send 0
            send_data = 0
            if best_combo.size != 0:  # some combo was good
                send_data = best_combo[0]

    return send_data


def run_on_trace(net_env, log_path):
    # past errors in bandwidth
    past_errors = []
    past_bandwidth_ests = []
    chunk_combo_options = []
    video_size = np.array([net_env.video_size[i]
                           for i in sorted(net_env.video_size)])
    with open(log_path, 'w', 1) as log_file:
        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        # a_batch = [action_vec]
        r_batch = []

        future_bandwidth = 0

        # make chunk combination options
        for combo in itertools.product(range(len(VIDEO_BIT_RATE)),
                                       repeat=MPC_FUTURE_CHUNK_COUNT):
            chunk_combo_options.append(combo)
        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max
        # reward combination
        # start = time.time()
        chunk_combo_options = np.array(chunk_combo_options)

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.step(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty
            reward = linear_reward(bit_rate, last_bit_rate, rebuf)

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\t' + str(future_bandwidth) + '\n')

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf
            state[3, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = video_chunk_remain / net_env.total_video_chunk
            # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

            # ================== MPC =========================
            # defualt assumes that this is the first request so error is 0
            # since we have never predicted bandwidth
            curr_error = 0
            if (len(past_bandwidth_ests) > 0):
                curr_error = abs(
                    past_bandwidth_ests[-1]-state[3, -1])/float(state[3, -1])
            past_errors.append(curr_error)

            # pick bitrate according to MPC
            # first get harmonic mean of last 5 bandwidths
            past_bandwidths = state[3, -5:]
            while past_bandwidths[0] == 0.0:
                past_bandwidths = past_bandwidths[1:]
            # if ( len(state) < 5 ):
            #    past_bandwidths = state[3,-len(state):]
            # else:
            #    past_bandwidths = state[3,-5:]
            bandwidth_sum = 0
            for past_val in past_bandwidths:
                bandwidth_sum += (1/float(past_val))
            harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

            # future bandwidth prediction
            # divide by 1 + max of last 5 (or up to 5) errors
            max_error = 0
            error_pos = -5
            if (len(past_errors) < 5):
                error_pos = -len(past_errors)
            max_error = float(max(past_errors[error_pos:]))
            future_bandwidth = harmonic_bandwidth / \
                (1+max_error)  # robustMPC here
            past_bandwidth_ests.append(harmonic_bandwidth)

            # future chunks length (try 4 if that many remaining)
            last_index = int(net_env.total_video_chunk -
                             video_chunk_remain) - 1
            future_chunk_length = min(MPC_FUTURE_CHUNK_COUNT,
                                      net_env.total_video_chunk-last_index-1)
            bit_rate = calculate_rebuffer(future_chunk_length, buffer_size,
                                          bit_rate, last_index,
                                          future_bandwidth,
                                          video_size,
                                          chunk_combo_options)

            s_batch.append(state)

            if end_of_video:
                break


def main():
    # args = parse_args()
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    np.random.seed(SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        TEST_TRACE_DIR)

    jobs = []
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(all_cooked_time, all_cooked_bw, all_file_names)):
        net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, SEED,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        log_path = os.path.join(SUMMARY_DIR, 'log_sim_mpc_' + trace_filename)
        p = mp.Process(target=run_on_trace, args=(net_env, log_path))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()


if __name__ == '__main__':
    t_start = time.time()
    main()
    print("time used: {}".format(time.time() - t_start))
