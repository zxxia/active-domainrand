"""Trace generator module."""
import numpy as np
import random


def transition(state, variance, prob_stay, bitrate_states, transition_probs,
               prng):
    """Hidden Markov State transition."""
    # variance_switch_prob, sigma_low, sigma_high,
    transition_prob = prng.uniform()

    if transition_prob < prob_stay:  # stay in current state
        return state
    else:  # pick appropriate state!
        # next_state = state
        curr_pos = state
        # first find max distance that you can be from current state
        max_distance = max(curr_pos, len(bitrate_states)-1-curr_pos)
        # cut the transition probabilities to only have possible number of
        # steps
        curr_transition_probs = transition_probs[0:max_distance]
        trans_sum = sum(curr_transition_probs)
        normalized_trans = [x/trans_sum for x in curr_transition_probs]
        # generate a random number and see which bin it falls in to
        trans_switch_val = prng.uniform()
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
        if (switch_down >= 0 and switch_up <= (len(bitrate_states)-1)):
            x = prng.uniform(0, 1)
            if (x < 0.5):
                return switch_up
            else:
                return switch_down
        elif switch_down >= 0:  # switch down
            return switch_down
        else:  # switch up
            return switch_up


class TraceGenerator(object):
    """Trace generator used to simulate a network trace."""

    def __init__(self, T_l, T_s, cov, duration, steps, min_throughput,
                 max_throughput, seed):
        """Construct a trace generator."""
        self.T_l = T_l
        self.T_s = T_s
        self.cov = cov
        self.duration = duration
        self.steps = steps
        self.min_throughput = min_throughput
        self.max_throughput = max_throughput

        # equivalent to Pensieve's way of computing switch parameter
        coeffs = np.ones(self.steps - 1)
        coeffs[0] = -1
        self.switch_parameter = np.real(np.roots(coeffs)[0])
        self.prng = np.random.RandomState(seed)

    def generate_trace(self):
        """Generate a network trace."""
        # get bitrate levels (in Mbps)
        bitrate_states = []
        curr = self.min_throughput
        for _ in range(0, self.steps):
            bitrate_states.append(curr)
            curr += (self.max_throughput-self.min_throughput)/(self.steps-1)

        # list of transition probabilities
        transition_probs = []
        # assume you can go steps-1 states away (we will normalize this to the
        # actual scenario)
        for z in range(1, self.steps-1):
            transition_probs.append(1/(self.switch_parameter**z))

        # probability to stay in same state
        prob_stay = 1 - 1 / self.T_l

        # takes a state and decides what the next state is

        current_state = int(self.prng.randint(0, len(bitrate_states)-1))
        current_variance = self.cov * bitrate_states[current_state]
        ts = 0
        cnt = 0
        trace_time = []
        trace_bw = []
        while ts < self.duration:
            # prints timestamp (in seconds) and throughput (in Mbits/s)
            if cnt <= 0:
                noise = self.prng.normal(0, current_variance, 1)[0]
                cnt = self.T_s
            # the gaussian val is at least 0.1
            gaus_val = max(0.1, bitrate_states[current_state] + noise)
            trace_time.append(ts)
            trace_bw.append(gaus_val)
            cnt -= 1
            next_val = transition(current_state, current_variance, prob_stay,
                                  bitrate_states, transition_probs, self.prng)
            if current_state != next_val:
                cnt = 0
            current_state = next_val
            current_variance = self.cov * bitrate_states[current_state]
            ts += 1
        return trace_time, trace_bw


class RandomTraceGenerator(object):
    """
    Trace generator used to simulate a network trace.
    non-MDP logic
    """

    def __init__(self, T_l, T_s, cov, duration, steps, min_throughput,
                 max_throughput, seed):
        """Construct a trace generator."""
        self.T_s = T_s
        self.duration = duration
        self.min_throughput = min_throughput
        self.max_throughput = max_throughput

    def generate_trace(self):
        """Generate a network trace."""
        round_digit = 2
        ts = 0
        cnt = 0
        trace_time = []
        trace_bw = []
        last_val = round( np.random.uniform( self.min_throughput, self.max_throughput ) ,round_digit )

        while trace_time < self.duration:
            if cnt <= 0:
                bw_val = round( np.random.uniform( self.min_throughput, self.max_throughput ) ,round_digit )
                cnt = np.random.randint( 1, self.T_s + 1 )

            elif cnt >= 1:
                bw_val = last_val
            else:
                bw_val = round( np.random.uniform( self.min_throughput, self.max_throughput ) ,round_digit )

            cnt -= 1
            ts = round( ts ,2 )

            last_val = bw_val
            time_noise = random.uniform( 0.1 ,3.5 )
            ts += time_noise
            trace_time.append( ts )
            trace_bw.append( bw_val )

        return trace_time, trace_bw
