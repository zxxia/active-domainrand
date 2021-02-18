"""Pensieve Environment Module."""

import json
import os
from typing import Union, List

import numpy as np

from pensieve.constants import (A_DIM, B_IN_MB, BITS_IN_BYTE,
                                BUFFER_NORM_FACTOR, DEFAULT_QUALITY, M_IN_K,
                                MILLISECONDS_IN_SECOND, NOISE_HIGH, NOISE_LOW,
                                S_INFO, S_LEN, VIDEO_BIT_RATE, VIDEO_CHUNK_LEN)
from pensieve.environment.dimension import Dimension
from pensieve.trace_generator import TraceGenerator
from pensieve.utils import linear_reward


class Environment:
    """Simulated network environment for Pensieve."""

    def __init__(self, video_size_file_dir: str, config_file: str, seed: int,
                 trace_time: Union[List[int], None] = None,
                 trace_bw: Union[List[float], None] = None,
                 trace_file_name: Union[str, None] = None, fixed: bool = True,
                 trace_video_same_duration_flag: bool =False):
        """Initialize a simulated network environment for Pensieve.

        Args
            video_size_file_dir(string): path to all video size files.
            config_file(string): path to all video size files.
            seed(int): random seed.
            trace_time(None or list): a list of timestamp in seconds.
            trace_time(None or list): a list of throughput in Mbits/second.
            trace_file_name(None or str): trace file name.
            fixed(boolean): if true, no random noises added to the delay.
            Random start of the trace should be handled outside environment.
            Default: True.
            trace_video_same_duration_flag(boolean): if true, total number of
            video chunks is the same as that in the video size file. Otherwise,
            total number of video chunks is aligned with the network trace.
            Default: false.
        """
        # variables related to video player and network configuration
        self.config_file = config_file
        self.seed = seed
        self.prng = np.random.RandomState(seed)

        self.dimensions = {}
        self._construct_dimensions()

        # variables related to network trace
        if trace_time is None and trace_bw is None:
            self.trace_generator = TraceGenerator(
                self.dimensions['T_l'].current_value,
                self.dimensions['T_s'].current_value,
                self.dimensions['cov'].current_value,
                self.dimensions['duration'].current_value,
                self.dimensions['step'].current_value,
                self.dimensions['min_throughput'].current_value,
                self.dimensions['max_throughput'].current_value, seed)
            self.trace_time, self.trace_bw = \
                self.trace_generator.generate_trace()
            self.trace_file_name = None
        elif trace_time is None and trace_bw is not None:
            raise ValueError("trace_time is None.")
        elif trace_time is not None and trace_bw is None:
            raise ValueError("trace_bw is None.")
        else:
            assert (isinstance(trace_time, List) and isinstance(trace_bw, List)
                    and len(trace_time) == len(trace_bw)), \
                "trace_bw and trace_time have different length."
            self.trace_time = trace_time
            self.trace_bw = trace_bw
            self.trace_generator = None
            self.trace_file_name = trace_file_name
        self.trace_video_same_duration_flag = trace_video_same_duration_flag

        # variables used to track the environment status
        self.trace_ptr = 1
        self.last_trace_ts = self.trace_time[self.trace_ptr - 1]
        self.nb_chunk_sent = 0
        self.buffer_size = 0
        self.last_bitrate = DEFAULT_QUALITY
        self.fixed = fixed
        self.state = np.zeros((1, S_INFO, S_LEN))

        # a dict mapping bitrate to vidoe chunk size in bytes
        self.video_size = {}
        self._construct_bitrate_chunksize_map(video_size_file_dir)

        # get total number of chunks in a video
        self.total_video_chunk = len(self.video_size[0]) # - 1
        if trace_video_same_duration_flag:
            # if the trace is longer than the video, extend the video
            self.total_video_chunk = max(
                self.total_video_chunk,
                self.trace_time[-1]*MILLISECONDS_IN_SECOND//VIDEO_CHUNK_LEN)

    def _construct_bitrate_chunksize_map(self, video_size_file_dir):
        """Construct a dict mapping bitrate to video chunk size."""
        self.video_size = {}  # in bytes
        for bitrate in range(len(VIDEO_BIT_RATE)):
            self.video_size[bitrate] = []
            video_size_file = os.path.join(video_size_file_dir,
                                           'video_size_{}'.format(bitrate))
            with open(video_size_file, 'r') as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def _construct_dimensions(self):
        """Load environment parameters' dimensions."""
        self.dimensions = {}

        with open(self.config_file, mode='r') as f:
            config = json.load(f)

        for dimension in config['dimensions']:
            self.dimensions[dimension['name']] = Dimension(
                default_value=dimension['default'],
                seed=self.seed,
                min_value=dimension['min'],
                max_value=dimension['max'],
                name=dimension['name'],
                unit=dimension['unit'])

    def close(self):
        """Do nothing.

        Placeholder function for elegant exit.
        """
        pass

    def randomize(self, randomized_values=None):
        """Create a randomized environment.

           Uniformly randomize when randomized_values is None. Otherwise
           directly assign randomized_values to environment parameters.

        Args
            randomized_values(dict or None): If None, do uniform randomization.
                If a dict mapping dimension name to a value, directly assign
                the value to the corresponding environment parameter.
        """
        regenerate_trace = False
        if randomized_values is None:
            for name, dim in self.dimensions.items():
                if dim.max_value != dim.min_value:  # need to random
                    if name == 'T_l' or name == 'T_s' or name == 'cov' or \
                            name == 'duration' or name == 'step' or \
                            name == 'min_throughput' or \
                            name == 'max_throughput':
                        regenerate_trace = True
                    dim.randomize()  # uniform randomization
        elif isinstance(randomized_values, dict):
            for name, dim in randomized_values.items():
                if name not in self.dimensions:
                    raise KeyError("Unrecoginized dimension, {}".format(name))
                new_val = randomized_values[name]
                # if new_val < self.dimensions[name].min_value or  \
                #         new_val > self.dimensions[name].max_value:
                #     raise ValueError(
                #         "New value {} is out of {}'s range [{}, {}].".format(
                #             new_val, name, self.dimensions[name].min_value,
                #             self.dimensions[name].max_value))
                self.dimensions[name].current_value = new_val
                if name == 'T_l' or name == 'T_s' or name == 'cov' or \
                        name == 'duration' or name == 'step' or \
                        name == 'min_throughput' or \
                        name == 'max_throughput':
                    regenerate_trace = True
        else:
            raise ValueError("Unrecoginized type of input.")

        if regenerate_trace:
            self.trace_generator.T_l = self.dimensions['T_l'].current_value
            self.trace_generator.T_s = self.dimensions['T_s'].current_value
            self.trace_generator.cov = self.dimensions['cov'].current_value
            self.trace_generator.duration = \
                self.dimensions['duration'].current_value
            self.trace_generator.steps = self.dimensions['step'].current_value
            self.trace_generator.min_throughput = \
                self.dimensions['min_throughput'].current_value
            self.trace_generator.max_throughput = \
                self.dimensions['max_throughput'].current_value
            self.trace_time, self.trace_bw = \
                self.trace_generator.generate_trace()
            # print('generate trace!', self.trace_generator.T_l,
            #       self.trace_generator.T_s, self.trace_generator.cov,
            #       self.trace_generator.duration, self.trace_generator.steps,
            #       self.trace_generator.min_throughput,
            #       self.trace_generator.max_throughput)
            self.reset()
            # TODO: may need to save the new network trace.

    def reset(self, **kwargs):
        """Reset the environment paramters to default values."""
        if 'random_start' in kwargs and kwargs['random_start']:
            self.trace_ptr = self.prng.randint(len(self.trace_time))
            self.last_trace_ts = self.trace_time[self.trace_ptr - 1]
        else:
            self.trace_ptr = 1
            self.last_trace_ts = self.trace_time[self.trace_ptr - 1]
        self.nb_chunk_sent = 0
        self.buffer_size = 0
        self.last_bitrate = DEFAULT_QUALITY
        self.state = np.zeros((1, S_INFO, S_LEN))

    def step(self, bitrate):
        """Step the environment by inputting an action.

        Read the network trace and transmit a video chunk. The funcition is
        renamed from get_video_chunk in old pensieve code to be compatible with
        the convention of ADR.
        """
        self.state = np.roll(self.state, -1, -1)
        link_rtt = self.dimensions['link_rtt'].current_value  # millisec
        buffer_thresh = self.dimensions['buffer_threshold'].current_value * \
            MILLISECONDS_IN_SECOND  # millisec, max buffer limit
        drain_buffer_sleep_time = self.dimensions[
            'drain_buffer_sleep_time'].current_value  # millisec
        packet_payload_portion = self.dimensions[
            'packet_payload_portion'].current_value

        assert 0 <= bitrate < len(VIDEO_BIT_RATE)

        video_chunk_size = self.video_size[bitrate][
            self.nb_chunk_sent % (len(self.video_size[bitrate]))]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        bytes_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            # throughput = bytes per ms
            throughput = self.trace_bw[self.trace_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = self.trace_time[self.trace_ptr] - self.last_trace_ts

            packet_payload = throughput * duration * packet_payload_portion

            if bytes_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - bytes_sent) / \
                    throughput / packet_payload_portion
                delay += fractional_time
                self.last_trace_ts += fractional_time
                assert self.last_trace_ts <= self.trace_time[self.trace_ptr]
                break

            bytes_sent += packet_payload
            delay += duration
            self.last_trace_ts = self.trace_time[self.trace_ptr]
            self.trace_ptr += 1

            if self.trace_ptr >= len(self.trace_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.trace_ptr = 1
                self.last_trace_ts = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += link_rtt

        # add a multiplicative noise to the delay
        if not self.fixed:
            delay *= self.prng.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > buffer_thresh:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - buffer_thresh
            sleep_time = drain_buffer_sleep_time * \
                np.ceil(drain_buffer_time / drain_buffer_sleep_time)
            self.buffer_size -= sleep_time

            while True:
                duration = self.trace_time[self.trace_ptr] - self.last_trace_ts
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_trace_ts += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_trace_ts = self.trace_time[self.trace_ptr]
                self.trace_ptr += 1

                if self.trace_ptr >= len(self.trace_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.trace_ptr = 1
                    self.last_trace_ts = self.trace_time[self.trace_ptr - 1]

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.nb_chunk_sent += 1
        video_chunk_remain = self.total_video_chunk - self.nb_chunk_sent

        # whether it’s time to reset the environment again.
        # (https://gym.openai.com/docs/#spaces)
        end_of_video = self.nb_chunk_sent >= self.total_video_chunk

        next_video_chunk_sizes = []
        for i in range(len(VIDEO_BIT_RATE)):
            next_video_chunk_sizes.append(
                self.video_size[i][
                    self.nb_chunk_sent % (len(self.video_size[i]))])

        rebuf = rebuf / MILLISECONDS_IN_SECOND

        reward = linear_reward(VIDEO_BIT_RATE[bitrate],
                               VIDEO_BIT_RATE[self.last_bitrate], rebuf)
        self.last_bitrate = bitrate

        self.state[0, 0, -1] = VIDEO_BIT_RATE[bitrate] / np.max(VIDEO_BIT_RATE)
        self.state[0, 1, -1] = return_buffer_size / MILLISECONDS_IN_SECOND / \
            BUFFER_NORM_FACTOR
        self.state[0, 2, -1] = video_chunk_size / delay / M_IN_K  # kbyte/ms
        self.state[0, 3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        self.state[0, 4, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K
        self.state[0, 5, -1] = video_chunk_remain / self.total_video_chunk

        debug_info = {'delay': delay,
                      'sleep_time': sleep_time,
                      'buffer_size': return_buffer_size/MILLISECONDS_IN_SECOND,
                      'rebuf': rebuf,
                      'video_chunk_size': video_chunk_size,
                      'next_video_chunk_sizes': next_video_chunk_sizes,
                      'video_chunk_remain': video_chunk_remain}

        return self.state, reward, end_of_video, debug_info

    def get_dimension_values(self):
        return {name: dim.current_value
                for name, dim in self.dimensions.items()}

    def get_dims_with_rand(self):
        return {name: dim for name, dim in self.dimensions.items()
                if dim.min_value != dim.max_value}
