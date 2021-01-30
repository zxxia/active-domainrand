#! /bin/bash

# immediately exit the bash if an error encountered
set -e

# run this script in the project root directory
video_size_file_dir=pensieve/data/video_sizes
# CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/randomize_env_parameters.json'
# CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/randomize_env_parameters1.json'
# CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/randomize_env_parameters2.json'
# CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/randomize_parameters.json'
# CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/rand_buff_thresh.json'
# VAL_CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/default.json'
# SEED=42


if [ $(hostname) = "farewell" ]; then
    summary_dir='/tank/zxxia/active-domainrand/pensieve_results'
    train_config_file='pensieve/config/randomize_network_params1.json'
    val_config_file='pensieve/config/default.json'
    val_trace_dir=pensieve/data/synthetic_traces/val_rand_network_params
    randomization_interval=1000
    udr_type=udr
    method_name=${udr_type}_${randomization_interval}_rand_interval
    exp_name='randomize_network_params_range1'
    python -m pensieve.train_pensieve \
        --video-size-file-dir ${video_size_file_dir} \
        --train-env-config ${train_config_file} \
        --val-env-config ${val_config_file} \
        --summary-dir ${summary_dir}/${exp_name}/${method_name} \
        --randomization ${udr_type} \
        --val-trace-dir ${val_trace_dir} \
        --randomization-interval ${randomization_interval} \
        --model-save-interval 200
        --total-epoch 50000
        # --train-trace-dir ${TRAIN_TRACE_DIR} \
        # --test-env-config ${CONFIG_FILE} \
        # --test-trace-dir ${TEST_TRACE_DIR} \
elif [ $(hostname) = "silver" ]; then
    echo "in silver"
elif [ $(hostname) = "loon" ]; then
    echo "in loon"
else
    echo "Do nothing"
fi
