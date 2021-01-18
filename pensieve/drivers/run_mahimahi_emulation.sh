#!/bin/bash
set -e

ROOT=/home/zxxia/active-domainrand
MM_DELAY=100
TRACE_FILE=${ROOT}/pensieve/data/synthetic_traces/test_large_range/trace800.txt
VIDEO_SIZE_DIR=${ROOT}/data/video_sizes
# ACTOR_PATH=${ROOT}/results/7_dims_rand_large_range_correct_rebuf_penalty/even_udr_1_rand_interval/actor_ep_50000.pth
ACTOR_PATH=${ROOT}/results/7_dims_rand/even_udr_1_rand_interval/actor_ep_50000.pth
UP_LINK_SPEED_FILE=${ROOT}/data/12mbps
TRACE_DIR=data/synthetic_traces/test_7_dim_rand_in_dist_mahimahi
TRACE_FILES=`ls ${TRACE_DIR}`

# The architecture of emulation experiment.

#     localhost                |                 mahimahi container(shell)
#                              |
#  HTTP File server <-----video data, bitrate -----> virtual browser (run html, javascript)
#                              |                     ^
#                              |                     |
#                              |           state, bitrate decision
#                              |                     |
#                              |                     V
#                              |               abr(RL, MPC) server

# cd ${ROOT}/pensieve/video_server
# python -m http.server &
# cd ${ROOT}

# for MM_DELAY in 5 50 500 5000 do
trap "pkill -f abr_server" SIGINT
trap "pkill -f abr_server" EXIT
# trap "pkill -f abr_server && pkill -f 'python -m http.server'" SIGINT
# trap "pkill -f abr_server && pkill -f 'python -m http.server'" EXIT
for MM_DELAY in 100; do
    for TRACE_FILE in ${TRACE_FILES} ; do
        # pkill -f "python -m http.server"

        sleep 5
        # mm-delay ${MM_DELAY} mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}/${TRACE_FILE} -- \
        #     bash -c "python -m pensieve.abr_server --abr RobustMPC \
        #                         --video-size-file-dir ${VIDEO_SIZE_DIR} \
        #                         --summary-dir pensieve/tests/mpc_test \
        #                         --trace-file ${TRACE_FILE} --actor-path ${ACTOR_PATH} &
        #             abr_server_pid=\$! &&
        #             python -m pensieve.virtual_browser --ip \${MAHIMAHI_BASE} --port 8000 --abr RL;
        #             kill \${abr_server_pid} && echo kill\${abr_server_pid}"
        mm-delay ${MM_DELAY} \
            mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}/${TRACE_FILE} -- \
            bash -c "python -m pensieve.virtual_browser.virtual_browser --ip \${MAHIMAHI_BASE} --port 8000 --abr RL \
                                --video-size-file-dir ${VIDEO_SIZE_DIR} \
                                --summary-dir pensieve/tests/RL_100 \
                                --trace-file ${TRACE_FILE} --actor-path ${ACTOR_PATH}"
        sleep 5
        mm-delay ${MM_DELAY} \
            mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}/${TRACE_FILE} -- \
            bash -c "python -m pensieve.virtual_browser.virtual_browser --ip \${MAHIMAHI_BASE} --port 8000 --abr RobustMPC \
                                --video-size-file-dir ${VIDEO_SIZE_DIR} \
                                --summary-dir pensieve/tests/mpc_100 \
                                --trace-file ${TRACE_FILE} --actor-path ${ACTOR_PATH}"
    done
done
