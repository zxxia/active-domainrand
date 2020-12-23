import logging

# import gym
import matplotlib
import numpy as np
import torch
from experiments.domainrand.args import check_args, get_args

from pensieve.agent_policy import Pensieve, RobustMPC
from pensieve.svpg_simulator_agent import SVPGSimulatorAgent

matplotlib.use('Agg')

CONFIG_FILE = '/data3/zxxia/active-domainrand/pensieve/config/default.json'
VIDEO_SIZE_FILE_DIR = '/data3/zxxia/pensieve/data/video_sizes'

if __name__ == '__main__':
    args = get_args()
    # paths = setup_experiment_logs(args)
    check_args(args)

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # stats_logger = StatsLogger(args)
    # visualizer = Visualizer(randomized_env_id=args.randomized_eval_env_id,
    # seed=args.seed)

    # TODO: fix the envionment loading
    reference_agent_policy = RobustMPC()

    if args.freeze_agent:
        # TODO: only need the actor
        agent_policy = Pensieve(
            2, '/data3/zxxia/active-domainrand/pensieve/tests/pensieve_log')
    else:
        agent_policy = Pensieve(
            2, '/data3/zxxia/active-domainrand/pensieve/tests/pensieve_log')

        # if args.load_agent:
        #     agent_policy.load_models()

    # simulator_agent = generate_simulator_agent(args)
    simulator_agent = SVPGSimulatorAgent(
        VIDEO_SIZE_FILE_DIR,
        reference_agent_policy,
        randomized_env_id=CONFIG_FILE,
        randomized_eval_env_id=CONFIG_FILE,
        agent_name='pensieve',
        nagents=args.nagents,
        nparams=args.nparams,
        temperature=args.temperature,
        svpg_rollout_length=args.svpg_rollout_length,
        svpg_horizon=args.svpg_horizon,
        max_step_length=args.max_step_length,
        reward_scale=args.reward_scale,
        initial_svpg_steps=args.initial_svpg_steps,
        max_env_timesteps=args.max_env_timesteps,
        episodes_per_instance=args.episodes_per_instance,
        discrete_svpg=args.discrete_svpg,
        load_discriminator=args.load_discriminator,
        freeze_discriminator=args.freeze_discriminator,
        freeze_agent=args.freeze_agent,
        seed=args.seed,
        particle_path=args.particle_path,
    )

    svpg_timesteps = 0

    while simulator_agent.agent_timesteps < args.max_agent_timesteps:

        logging.info("SVPG TS: {}, Agent TS: {}".format(
            svpg_timesteps, simulator_agent.agent_timesteps))

        solved, info = simulator_agent.select_action(agent_policy)
        svpg_timesteps += 1
