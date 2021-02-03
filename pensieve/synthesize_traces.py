import argparse
import csv
import logging
import os
import random
import glob
import time

import numpy as np
from sympy import N, Symbol, solve
from pensieve.utils import read_json_file


def parse_args():
    '''
    Parse arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Generate synthetic data.")
    parser.add_argument("--config-file", type=str, required=True, help="config"
                        " of trace generator.")
    # parser.add_argument("--T_l-range", nargs=2, type=float, required=True,
    #                     help='control the prob_stay')
    # parser.add_argument("--T_s-range", nargs=2, type=float, required=True,
    #                     help='control how long to recompute noise')
    # parser.add_argument("--cov-range", nargs=2, type=float, required=True,
    #                     help='coefficient used to compute vairance of a state')
    # parser.add_argument("--duration-range", nargs=2, type=float, required=True,
    #                     help='duration of each synthetic trace in seconds.')
    # parser.add_argument("--steps", type=int, default=10,
    #                     help='number of steps')
    # parser.add_argument("--throughput-range", nargs=2, type=float,
    #                     default=[0.2, 4.3], help='range of throughput(Mbps)')
    parser.add_argument("--num-traces", type=int, required=True, help="number"
                        " of traces to be generated.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help='Output file directory.')

    return parser.parse_args()


def check_args(args):
    assert args.T_l_range[0] <= args.T_l_range[1]
    assert args.T_s_range[0] <= args.T_s_range[1]
    assert args.cov_range[0] <= args.cov_range[1]
    assert args.duration_range[0] <= args.duration_range[1]
    assert args.throughput_range[0] <= args.throughput_range[1]


def log_args(args):
    """
    Writes arguments to log. Assumes args.results_dir exists.
    """
    log_file = os.path.join(args.output_dir, '.log_args')
    args_logging = logging.getLogger("args")
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    args_logging.setLevel(logging.INFO)
    args_logging.addHandler(file_handler)
    args_logging.addHandler(stream_handler)
    for arg in vars(args):
        args_logging.info(arg + '\t' + str(getattr(args, arg)))


def transition(state, variance, prob_stay, bitrate_states_low_var,
               transition_probs):
    # variance_switch_prob, sigma_low, sigma_high,
    transition_prob = random.uniform(0, 1)

    if transition_prob < prob_stay:  # stay in current state
        return state
    else:  # pick appropriate state!
        # next_state = state
        curr_pos = state
        # first find max distance that you can be from current state
        max_distance = max(curr_pos, len(bitrate_states_low_var)-1-curr_pos)
        # cut the transition probabilities to only have possible number of
        # steps
        curr_transition_probs = transition_probs[0:max_distance]
        trans_sum = sum(curr_transition_probs)
        normalized_trans = [x/trans_sum for x in curr_transition_probs]
        # generate a random number and see which bin it falls in to
        trans_switch_val = random.uniform(0, 1)
        running_sum = 0
        num_switches = -1
        for ind in range(0, len(normalized_trans)):
            # this is the val
            if (trans_switch_val <= (normalized_trans[ind] + running_sum)):
                num_switches = ind
                break
            else:
                running_sum += normalized_trans[ind]

        # now check if there are multiple ways to move this many states away
        switch_up = curr_pos + num_switches
        switch_down = curr_pos - num_switches
        # can go either way
        if (switch_down >= 0 and switch_up <= (len(bitrate_states_low_var)-1)):
            x = random.uniform(0, 1)
            if (x < 0.5):
                return switch_up
            else:
                return switch_down
        elif switch_down >= 0:  # switch down
            return switch_down
        else:  # switch up
            return switch_up


def generate_trace(T_s, T_l, cov, time_length, steps, max_throughput,
                   min_throughput, switch_parameter, output_file):
    # get bitrate levels (in Mbps)
    bitrate_states_low_var = []
    curr = min_throughput
    for _ in range(0, steps):
        bitrate_states_low_var.append(curr)
        curr += ((max_throughput - min_throughput)/(steps - 1))

    # list of transition probabilities
    transition_probs = []
    # assume you can go steps-1 states away (we will normalize this to the
    # actual scenario)
    for z in range(1, steps-1):
        transition_probs.append(1/(switch_parameter**z))

    # probability to stay in same state
    prob_stay = 1 - 1 / T_l

    # takes a state and decides what the next state is

    current_state = random.randint(0, len(bitrate_states_low_var)-1)
    current_variance = cov * bitrate_states_low_var[current_state]
    ts = 0
    cnt = 0
    with open(output_file, 'w', 1) as f:
        output_writer = csv.writer(f, delimiter='\t')
        while ts < time_length:
            # prints timestamp (in seconds) and throughput (in Mbits/s)
            if cnt <= 0:
                noise = np.random.normal(0, current_variance, 1)[0]
                cnt = T_s
            # TODO: the gaussian val is at least 0.1
            gaus_val = max(0.1, bitrate_states_low_var[current_state] + noise)
            output_writer.writerow([ts, gaus_val, cnt])
            cnt -= 1
            next_val = transition(current_state, current_variance, prob_stay,
                                  bitrate_states_low_var, transition_probs)
            if current_state != next_val:
                cnt = 0
            current_state = next_val
            current_variance = cov * bitrate_states_low_var[current_state]
            ts += 1


def read_config_ranges(config_file):
    config_ranges = {}
    for dim in read_json_file(config_file)["dimensions"]:
        if dim["name"] in ["T_l", "T_s", "cov", "duration", "step",
                           "min_throughput", "max_throughput"]:
            config_ranges[dim["name"]] = [dim["min"], dim["max"]]
    return config_ranges


def main():
    args = parse_args()
    config_ranges = read_config_ranges(args.config_file)
    # check_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    log_args(args)
    t_start = time.time()
    eq = -1
    x = Symbol("x", positive=True)
    steps = random.randint(*config_ranges["step"])
    print(steps)
    for y in range(1, steps-1):
        eq += (1/x**y)
    res = solve(eq, x)
    switch_parameter = N(res[0])
    print('solving polynomial: {:.3f}s'.format(time.time()-t_start))
    if os.path.exists(os.path.join(args.output_dir, ".metadata.csv")):
        f = open(os.path.join(args.output_dir, ".metadata.csv"), 'a', 1)
        csvwriter = csv.writer(f)
    else:
        f = open(os.path.join(args.output_dir, ".metadata.csv"), 'w', 1)
        csvwriter = csv.writer(f)
        csvwriter.writerow(['filename', 'T_s', 'T_l', "cov", 'duration',
                            'steps', 'max_throughput', 'min_throughput',
                            'switch_parameter'])
    existed_traces = glob.glob(os.path.join(args.output_dir, "trace*.txt"))
    for i in range(args.num_traces):
        output_file = os.path.join(args.output_dir,
                                   "trace{}.txt".format(len(existed_traces)+i))
        T_s = random.uniform(*config_ranges["T_s"])
        T_l = random.uniform(*config_ranges["T_l"])
        cov = random.uniform(*config_ranges["cov"])
        duration = random.uniform(*config_ranges["duration"])
        # min_throughput, max_throughput = args.throughput_range
        min_throughput = random.uniform(*config_ranges["min_throughput"])
        max_throughput = random.uniform(*config_ranges["max_throughput"])

        generate_trace(T_s, T_l, cov, duration, steps, max_throughput,
                       min_throughput, switch_parameter, output_file)
        csvwriter.writerow([os.path.basename(output_file), T_s, T_l, cov,
                            duration, steps, max_throughput,
                            min_throughput, switch_parameter])
    print('e2e: {:.3f}s'.format(time.time()-t_start))


if __name__ == "__main__":
    main()
