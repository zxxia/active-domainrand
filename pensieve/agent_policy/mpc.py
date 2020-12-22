import itertools
import multiprocessing as mp
import os
import time

import numpy as np
from numba import jit

from pensieve.agent_policy import BaseAgentPolicy
from pensieve.constants import (A_DIM, B_IN_MB, DEFAULT_QUALITY, M_IN_K,
                                REBUF_PENALTY, S_INFO, S_LEN, SMOOTH_PENALTY,
                                TOTAL_VIDEO_CHUNK, VIDEO_BIT_RATE)
from pensieve.environment import Environment

VIDEO_BIT_RATE = np.array(VIDEO_BIT_RATE)  # Kbps


class RobustMPC:
    """Naive implementation of RobustMPC."""

    def __init__(self, mpc_future_chunk_cnt=5):
        self.mpc_future_chunk_cnt = mpc_future_chunk_cnt

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max
        # reward combination
        self.chunk_combo_options = np.array(
            [combo for combo in itertools.product(
                range(len(VIDEO_BIT_RATE)), repeat=self.mpc_future_chunk_cnt)])
        raise NotImplementedError

    def evaluate(self, net_env):
        """Evaluate on a single net_env."""
        net_env.reset()
        # past errors in bandwidth
        past_errors = []
        past_bandwidth_ests = []
        video_size = np.array([net_env.video_size[i]
                               for i in sorted(net_env.video_size)])
        time_stamp = 0

        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        # a_batch = [action_vec]
        r_batch = []

        future_bandwidth = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.step(bit_rate)

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms

            r_batch.append(reward)

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
            future_bandwidth = harmonic_bandwidth / (1+max_error)  # robustMPC
            past_bandwidth_ests.append(harmonic_bandwidth)

            # future chunks length (try 4 if that many remaining)
            last_index = int(net_env.total_video_chunk -
                             info['video_chunk_remain']) - 1
            future_chunk_length = min(self.mpc_future_chunk_cnt,
                                      net_env.total_video_chunk-last_index-1)
            # TODO: refactor this into select action
            bit_rate = calculate_rebuffer(
                future_chunk_length, info['buffer_size'], bit_rate, last_index,
                future_bandwidth, video_size, self.chunk_combo_options)

            if end_of_video:
                break
        raise NotImplementedError

    def test_envs(self, net_envs):
        """Evaluate MultiEnv"""
        jobs = []
        for net_env in net_envs.net_envs:
            p = mp.Process(target=self.evaluate, args=(net_env))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
            raise NotImplementedError


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
