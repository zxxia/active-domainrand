"""Pensieve Environment Module."""

import json
import os

import numpy as np
from constants import (B_IN_MB, BITS_IN_BYTE, MILLISECONDS_IN_SECOND,
                       NOISE_HIGH, NOISE_LOW, VIDEO_BIT_RATE, VIDEO_CHUNK_LEN)
from dimension import Dimension
from trace_generator import TraceGenerator


class Environment(object):
    """Simulated network environment for Pensieve."""

    def __init__(self, video_size_file_dir, config_file, seed, trace_time=None,
                 trace_bw=None, trace_file_name=None, fixed=True,
                 trace_video_same_duration_flag=False):
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
                self.dimensions['max_throughput'].current_value)
            self.trace_time, self.trace_bw = \
                self.trace_generator.generate_trace()
            self.trace_file_name = None
        elif trace_time is None and trace_bw is not None:
            raise ValueError("trace_time is None.")
        elif trace_time is not None and trace_bw is None:
            raise ValueError("trace_bw is None.")
        else:
            assert len(trace_time) == len(trace_bw), \
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
        self.fixed = fixed

        # a dict mapping bitrate to vidoe chunk size in bytes
        self.video_size = {}
        self._construct_bitrate_chunksize_map(video_size_file_dir)

        # get total number of chunks in a video
        self.total_video_chunk = len(self.video_size[0])
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

    def randomize(self, randomized_values=None):
        """Create a randomized environment.

           Uniformly randomize when randomized_values is None. Otherwise
           directly assign randomized_values to environment parameters.

        Args
            randomized_values(float): If not None, directly assign this value
            to the environment parameters. If None, do uniform randomization.
        """
        # TODO: need to implement randomized_values assignment.
        for dimension in self.dimensions:
            dimension.randomize()
        # TODO: need to consider whether to create a new network trace.

    def reset(self, **kwargs):
        """Reset the environment paramters to default values."""
        # TODO: reset the environment paramters.
        raise NotImplementedError

    def step(self, quality):
        """Step the environment by inputting an action.

        Read the network trace and transmit a video chunk. The funcition is
        renamed from get_video_chunk in old pensieve code to be compatible with
        the convention of ADR.
        """
        link_rtt = self.dimensions['link_rtt'].current_value  # millisec
        buffer_thresh = self.dimensions['buffer_thresh'].current_value * \
            MILLISECONDS_IN_SECOND  # millisec, max buffer limit
        drain_buffer_sleep_time = self.dimensions[
            'drain_buffer_sleep_time'].current_value  # millisec
        packet_payload_portion = self.dimensions[
            'packet_payload_portion'].current_value

        assert 0 <= quality < len(VIDEO_BIT_RATE)

        video_chunk_size = self.video_size[quality][
            self.nb_chunk_sent % len(self.video_size[quality])]

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
                assert(self.last_trace_ts <= self.trace_time[self.trace_ptr])
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
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

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

        end_of_video = self.nb_chunk_sent >= self.total_video_chunk

        next_video_chunk_sizes = []
        for i in range(len(VIDEO_BIT_RATE)):
            next_video_chunk_sizes.append(
                self.video_size[i][
                    self.nb_chunk_sent % len(self.video_size[0])])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain
