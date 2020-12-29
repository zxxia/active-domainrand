
import math
import os
import numpy as np
import random

from numba import jit
from pensieve.constants import (VIDEO_BIT_RATE, HD_REWARD, SMOOTH_PENALTY,
                                REBUF_PENALTY, M_IN_K)

NAMES = ['timestamp', 'bandwidth']


def load_traces(trace_dir):
    trace_files = sorted(os.listdir(trace_dir))
    all_ts = []
    all_bw = []
    all_file_names = []
    for trace_file in trace_files:
        # if trace_file in ["ferry.nesoddtangen-oslo-report.2011-02-01_1000CET.log", "trace_32551_http---www.amazon.com", "trace_5294_http---www.youtube.com", "trace_5642_http---www.youtube.com"]:
        #     continue
        if trace_file.startswith("."):
            continue
        file_path = os.path.join(trace_dir, trace_file)
        ts_list = []
        bw_list = []
        with open(file_path, 'r') as f:
            for line in f:
                if len(line.split()) > 2:
                    ts, bw, _ = line.split()
                else:
                    ts, bw = line.split()
                ts = float(ts)
                bw = float(bw)
                ts_list.append(ts)
                bw_list.append(bw)
        all_ts.append(ts_list)
        all_bw.append(bw_list)
        all_file_names.append(trace_file)

    return all_ts, all_bw, all_file_names


def adjust_traces(all_ts, all_bw, bw_noise=0, duration_factor=1):
    new_all_bw = []
    new_all_ts = []
    for trace_ts, trace_bw in zip(all_ts, all_bw):
        duration = trace_ts[-1]
        new_duration = duration_factor * duration
        new_trace_ts = []
        new_trace_bw = []
        for i in range(math.ceil(duration_factor)):
            for t, bw in zip(trace_ts, trace_bw):
                if (t + i * duration) <= new_duration:
                    new_trace_ts.append(t + i * duration)
                    new_trace_bw.append(bw+bw_noise)

        new_all_ts.append(new_trace_ts)
        new_all_bw.append(new_trace_bw)
    assert len(new_all_ts) == len(all_ts)
    assert len(new_all_bw) == len(all_bw)
    return new_all_ts, new_all_bw


def compute_cdf(data):
    """ Return the cdf of input data.

    Args
        data(list): a list of numbers.

    Return
        sorted_data(list): sorted list of numbers.

    """
    length = len(data)
    sorted_data = sorted(data)
    cdf = [i / length for i, val in enumerate(sorted_data)]
    return sorted_data, cdf


def adjust_traces_one_random(all_ts, all_bw, random_seed, robust_noise, sample_length):
    adjust_n_random_traces(all_ts, all_bw, random_seed,
                           robust_noise, sample_length, number_pick=1)
    # new_all_bw = all_bw.copy()
    # new_all_ts = all_ts.copy()
    # np.random.seed(random_seed)
    #
    # number_of_traces = len(all_ts)
    # random_trace_index = random.randint(0, number_of_traces - 1)
    # trace_bw = new_all_bw[random_trace_index]
    #
    # ########
    # # use your randomization code from the notebook on new_all_bw
    # ########
    # start_index = random.randint( 0, len( trace_bw ) - sample_length )
    # sublist = trace_bw[start_index: start_index + sample_length]
    # trace_bw[start_index:start_index + sample_length] = [i * float(1+robust_noise) for i in sublist]
    #
    # assert len(new_all_ts) == len(all_ts)
    # assert len(new_all_bw) == len(all_bw)
    #
    # return new_all_ts, new_all_bw


def adjust_n_random_traces(all_ts, all_bw, random_seed, robust_noise, sample_length, number_pick):
    new_all_bw = all_bw.copy()
    new_all_ts = all_ts.copy()
    random.seed(random_seed)
    np.random.seed(random_seed)

    number_of_traces = len(all_ts)

    # we need n random index numbers from the set
    # do this n times
    random_trace_indices = random.sample(
        range(0, number_of_traces - 1), number_pick)

    for ri in random_trace_indices:
        trace_bw = new_all_bw[ri]

        start_index = random.randint(0, len(trace_bw) - sample_length)
        sublist = trace_bw[start_index: start_index + sample_length]
        new_sublist = []
        for i in sublist:
            # add constant noise
            # i = i*float(1+robust_noise)
            # if i + robust_noise > 0:
            #     i = i + robust_noise
            # else:
            #     i = i
            # new_sublist.append(i)

            # add normal noise
            noise = np.random.normal(0, 0.1, 1)
            if noise < -0.5 or noise > 0.5:
                noise = 0
            delta = 1 + float(noise)
            new_sublist.append(i * delta)

        trace_bw[start_index:start_index + sample_length] = new_sublist

    assert len(new_all_ts) == len(all_ts)
    assert len(new_all_bw) == len(all_bw)

    return new_all_ts, new_all_bw


@jit(nopython=True)
def linear_reward(current_bitrate, last_bitrate, rebuffer):
    reward = current_bitrate / M_IN_K - REBUF_PENALTY * rebuffer - \
        SMOOTH_PENALTY * np.abs(current_bitrate - last_bitrate) / M_IN_K
    return reward


def opposite_linear_reward(current_bitrate_idx, last_bitrate_idx, rebuffer):
    """Return linear reward which encourages rebuffering and bitrate switching.
    """
    current_bitrate = VIDEO_BIT_RATE[current_bitrate_idx]
    last_bitrate = VIDEO_BIT_RATE[last_bitrate_idx]
    reward = current_bitrate / M_IN_K + REBUF_PENALTY * rebuffer + \
        SMOOTH_PENALTY * np.abs(current_bitrate - last_bitrate) / M_IN_K
    return reward


def log_scale_reward(current_bitrate_idx, last_bitrate_idx, rebuffer):
    current_bitrate = VIDEO_BIT_RATE[current_bitrate_idx]
    last_bitrate = VIDEO_BIT_RATE[last_bitrate_idx]
    log_bit_rate = np.log(current_bitrate / VIDEO_BIT_RATE[-1])
    log_last_bit_rate = np.log(last_bitrate / VIDEO_BIT_RATE[-1])
    reward = log_bit_rate - REBUF_PENALTY * rebuffer - SMOOTH_PENALTY * \
        np.abs(log_bit_rate - log_last_bit_rate)
    return reward


def hd_reward(current_bitrate_idx, last_bitrate_idx, rebuffer):
    reward = HD_REWARD[current_bitrate_idx] - \
        REBUF_PENALTY * rebuffer - SMOOTH_PENALTY * \
        np.abs(HD_REWARD[current_bitrate_idx] - HD_REWARD[last_bitrate_idx])
    return reward


def evaluate_policy(nagents, net_envs, agent_policy,  # replay_buffer,
                    eval_episodes, max_steps, freeze_agent=True,
                    return_rewards=False,
                    add_noise=False, log_distances=True,  # gail_rewarder=None,
                    noise_scale=0.1, min_buffer_len=1000):
    """Evaluate a given policy in a set of environments.

    Return an array of rewards received from the evaluation step.
    """
    # TODO: environemnts have the same random seed. Need to fix, consider
    # multiproc
    # warning: runnable but may have logic errors Double check the logic.
    assert nagents == len(net_envs)
    states_list = [[] for _ in range(nagents)]
    actions_list = [[] for _ in range(nagents)]
    next_states_list = [[] for _ in range(nagents)]
    rewards_list = [[] for _ in range(nagents)]
    ep_rewards = []
    final_dists = []

    for ep in range(eval_episodes):
        agent_total_rewards = np.zeros(nagents)
        states = []
        rewards = []
        actions = []
        dones = []
        infos = []
        for net_env in net_envs:
            net_env.reset()
            state, reward, done, info = net_env.step(1)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        # done = [False] * nagents
        add_to_buffer = [True] * nagents
        steps = 0
        training_iters = 0

        while not all(dones) and steps <= max_steps:
            actions = []
            for state in states:
                actions.append(agent_policy.select_action(state))

            next_states = []
            rewards = []
            dones = []
            infos = []
            for net_env, action in zip(net_envs, actions):
                next_state, reward, done, info = net_env.step(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

            for i, st in enumerate(states):
                if add_to_buffer[i]:
                    states_list[i].append(st)
                    actions_list[i].append(actions[i])
                    next_states_list[i].append(next_states[i])
                    rewards_list[i].append(rewards[i])
                    agent_total_rewards[i] += rewards[i]
                    training_iters += 1

                    # if replay_buffer is not None:
                    #     done_bool = 0 if steps + \
                    #         1 == max_steps else float(done[i])
                    #     replay_buffer.add(
                    #         (state[i], next_state[i], action[i], reward[i], done_bool))

                if dones[i]:
                    # Avoid duplicates
                    add_to_buffer[i] = False

                    if log_distances:
                        final_dists.append(info[i]['goal_dist'])

            state = next_state
            steps += 1

        # Train for total number of env iterations
        # and len(replay_buffer.storage) > min_buffer_len:
        if not freeze_agent:
            # agent_policy.train(replay_buffer=replay_buffer,
            #                    iterations=training_iters)
            agent_policy.train(net_envs, iters=training_iters)

        ep_rewards.append(agent_total_rewards)

    if return_rewards:
        return np.array(ep_rewards).flatten(), np.array(final_dists).flatten()

    trajectories = []
    for i in range(nagents):
        # print(np.array(states_list[i])[:, 0, :, -1].shape)
        # print(np.array(actions_list[i]).reshape((-1, 1)).shape)
        # print(np.array(next_states_list[i])[:, 0, :, -1].shape)
        # TODO: what states to use here
        # pensive models take a matrix as state
        # here only takes the latest state
        trajectories.append(np.concatenate(
            [
                np.array(states_list[i])[:, 0, :, -1],
                np.array(actions_list[i]).reshape((-1, 1)),
                np.array(next_states_list[i])[:, 0, :, -1]
            ], axis=-1))

    return trajectories
