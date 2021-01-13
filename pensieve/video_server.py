
import argparse
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
import time

import numpy as np

from pensieve.agent_policy import Pensieve, RobustMPC
from pensieve.constants import (
    A_DIM,
    BUFFER_NORM_FACTOR,
    DEFAULT_QUALITY,
    M_IN_K,
    S_INFO,
    S_LEN,
    TOTAL_VIDEO_CHUNK,
    VIDEO_BIT_RATE,
)
from pensieve.utils import construct_bitrate_chunksize_map, linear_reward

RANDOM_SEED = 42
RAND_RANGE = 1000


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Video Server")
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description of the experiment.')
    # ABR related
    parser.add_argument('--abr', type=str, required=True,
                        choices=['RobustMPC', 'RL'],
                        help='ABR algorithm.')
    parser.add_argument('--nn_model', type=str, default=None,
                        help='Path to RL model.')
    # data io related
    parser.add_argument('--save_dir', type=str, help='directory to save logs.')
    parser.add_argument('--trace_file', type=str, help='Path to trace file.')
    parser.add_argument("--video-size-file-dir", type=str, required=True,
                        help='Dir to video size files')

    # networking related
    parser.add_argument('--ip', type=str, default='localhost',
                        help='ip address of ABR/video server.')
    parser.add_argument('--port', type=int, default=8333,
                        help='port number of ABR/video server.')

    return parser.parse_args()


def make_request_handler(abr, log_file_path, video_size):
    """Instantiate HTTP request handler."""

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.log_writer = csv.writer(open(log_file_path, 'w', 1),
                                         delimiter='\t',
                                         lineterminator='\n')
            self.log_writer.writerow(
                ['wall_time', 'bit_rate', 'buffer_size', 'rebuffer_time',
                 'video_chunk_size', 'download_time', 'reward'])
            self.abr = abr
            self.video_chunk_count = 0
            self.last_total_rebuf = 0
            self.last_bit_rate = DEFAULT_QUALITY
            self.state = np.zeros((1, S_INFO, S_LEN))
            self.video_size = video_size

            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(
                content_length).decode('utf-8'))
            print(post_data)

            if ('pastThroughput' in post_data):
                # @Hongzi: this is just the summary of throughput/quality at
                # the end of the load so we don't want to use this information
                # to send back a new quality
                print("Summary: ", post_data)
            else:
                # option 1. reward for just quality
                # reward = post_data['lastquality']
                # option 2. combine reward for quality and rebuffer time
                #           tune up the knob on rebuf to prevent it more
                # reward = post_data['lastquality'] - 0.1 *
                # (post_data['RebufferTime'] -
                # self.input_dict['last_total_rebuf'])
                # option 3. give a fixed penalty if video is stalled
                #           this can reduce the variance in reward signal
                # reward = post_data['lastquality'] - 10 *
                # ((post_data['RebufferTime'] -
                # self.input_dict['last_total_rebuf']) > 0)

                # option 4. use the metric in SIGCOMM MPC paper
                rebuffer_time = float(
                    post_data['RebufferTime'] - self.last_total_rebuf)

                # --linear reward--
                reward = linear_reward(
                    VIDEO_BIT_RATE[post_data['lastquality']],
                    VIDEO_BIT_RATE[self.last_bit_rate], rebuffer_time)
                # VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                #     - REBUF_PENALTY * rebuffer_time / M_IN_K \
                #     - SMOOTH_PENALTY * np.abs(
                #     VIDEO_BIT_RATE[post_data['lastquality']] -
                #     self.last_bit_rate) / M_IN_K

                self.last_bit_rate = post_data['lastquality']
                self.last_total_rebuf = post_data['RebufferTime']

                # compute bandwidth measurement
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - \
                    post_data['lastChunkStartTime']
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                video_chunk_remain = TOTAL_VIDEO_CHUNK - \
                    self.video_chunk_count
                self.video_chunk_count += 1

                # dequeue history record
                self.state = np.roll(self.state, -1, -1)

                next_video_chunk_sizes = []
                for i in range(A_DIM):
                    next_video_chunk_sizes.append(
                        self.video_size[i][self.video_chunk_count])

                # this should be S_INFO number of terms
                # try:
                self.state[0, 0, -1] = VIDEO_BIT_RATE[post_data['lastquality']
                                                      ] / max(VIDEO_BIT_RATE)
                self.state[0, 1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR
                # kilo byte / ms
                self.state[0, 2, -1] = float(video_chunk_size) / \
                    float(video_chunk_fetch_time) / M_IN_K
                self.state[0, 3, -1] = float(video_chunk_fetch_time) / \
                    M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                self.state[0, 4, :A_DIM] = np.array(
                    next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                self.state[0, 5, -1] = min(
                    video_chunk_remain, TOTAL_VIDEO_CHUNK) / TOTAL_VIDEO_CHUNK
                # except ZeroDivisionError:
                #     # this should occur VERY rarely (1 out of 3000), should be
                #     # a dash issue in this case we ignore the observation and
                #     # roll back to an eariler one
                #     if len(self.s_batch) == 0:
                #         state = [np.zeros((S_INFO, S_LEN))]
                #     else:
                #         state = np.array(self.s_batch[-1], copy=True)

                # log wall_time, bit_rate, buffer_size, rebuffer_time,
                # video_chunk_size, download_time, reward
                self.log_writer.writerow(
                    [time.time(), VIDEO_BIT_RATE[post_data['lastquality']],
                     post_data['buffer'], rebuffer_time / M_IN_K,
                     video_chunk_size, video_chunk_fetch_time, reward])

                bit_rate, _ = self.abr.select_action(self.state)
                bit_rate = bit_rate.item()
                # action_prob = self.actor.predict(
                #     np.reshape(state, (1, S_INFO, S_LEN)))
                # action_cumsum = np.cumsum(action_prob)
                # bit_rate = (action_cumsum > np.random.randint(
                #     1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE
                # steps, because there is an intrinsic discrepancy in passing
                # single state and batch states

                # send data to html side
                send_data = str(bit_rate)

                end_of_video = post_data['lastRequest'] == TOTAL_VIDEO_CHUNK
                if end_of_video:
                    send_data = "REFRESH"
                    send_data = "STOP"
                    self.last_total_rebuf = 0
                    self.last_bit_rate = DEFAULT_QUALITY
                    self.video_chunk_count = 0
                    self.state = np.zeros((1, S_INFO, S_LEN))
                    # so that in the log we know where video ends
                    # self.log_writer.writerow('\n')

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(send_data)))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data.encode())

                # record [state, action, reward]
                # put it here after training, notice there is a shift in reward
                # storage

        def do_GET(self):
            print('GOT REQ')
            self.send_response(200)
            # self.send_header('Cache-Control', 'Cache-Control: no-cache,
            # no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', '20')
            self.end_headers()
            self.wfile.write(b"console.log('here');")

        def log_message(self, format, *args):
            return

    return Request_Handler


def main():
    args = parse_args()
    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)

    log_file_path = os.path.join(
        args.save_dir, 'log_RL_{}'.format(os.path.basename(args.trace_file)))

    ip = args.ip
    port = args.port

    if args.abr == 'RobustMPC':
        abr = RobustMPC()
    elif args.abr == 'RL':
        assert args.nn_model is not None, "nn_model is needed for RL abr."
        abr = Pensieve(16, args.summary_dir, actor_path=args.nn_model)
    else:
        raise ValueError("ABR {} is not supported!".format(args.abr))

    video_size = construct_bitrate_chunksize_map(args.video_size_file_dir)
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # interface to abr_rl server
    handler_class = make_request_handler(abr, log_file_path, video_size)

    server_address = (ip, port)
    httpd = HTTPServer(server_address, handler_class)
    print('Listening on ({}, {})'.format(ip, port))
    httpd.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
