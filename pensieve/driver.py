import logging

# import gym
import matplotlib
import numpy as np
import torch
# from common.utils.logging import (StatsLogger, reshow_hyperparameters,
#                                   setup_experiment_logs)
# from common.utils.sim_agent_helper import generate_simulator_agent
# from common.utils.visualization import Visualizer
from experiments.domainrand.args import check_args, get_args

from pensieve.pensieve import Pensieve
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
    # reference_env = gym.make(args.reference_env_id)

    if args.freeze_agent:
        # TODO: only need the actor
        agent_policy = Pensieve(
            2, '/data3/zxxia/active-domainrand/pensieve/tests/pensieve_log')
    else:
        agent_policy = Pensieve(
            2, '/data3/zxxia/active-domainrand/pensieve/tests/pensieve_log')

        if args.load_agent:
            agent_policy.load_model()

    # simulator_agent = generate_simulator_agent(args)
    simulator_agent = SVPGSimulatorAgent(
        VIDEO_SIZE_FILE_DIR,
        reference_env_id=CONFIG_FILE,
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
        # if svpg_timesteps % args.plot_frequency == 0:
        #     generalization_metric = visualizer.generate_ground_truth(simulator_agent, agent_policy, svpg_timesteps,
        #         log_path=paths['groundtruth_logs'])
        #
        #     np.savez('{}/generalization-seed{}.npz'.format(paths['paper'], args.seed),
        #         generalization_metric=generalization_metric,
        #         svpg_timesteps=svpg_timesteps,
        #         learning_curve_timesteps=simulator_agent.agent_timesteps
        #     )
        #
        #     visualizer.plot_reward(simulator_agent, agent_policy,
        #         svpg_timesteps, log_path=paths['policy_logs'], plot_path=paths['policy_plots'])
        #     visualizer.plot_value(simulator_agent, agent_policy,
        #         svpg_timesteps, log_path=paths['policy_logs'], plot_path=paths['policy_plots'])
        #     visualizer.plot_discriminator_reward(simulator_agent, agent_policy,
        #         svpg_timesteps, log_path=paths['policy_logs'], plot_path=paths['policy_plots'])
        #
        #     if not args.freeze_svpg:
        #         visualizer.plot_sampling_frequency(simulator_agent, agent_policy,
        #             svpg_timesteps, log_path=paths['sampling_logs'], plot_path=paths['sampling_plots'])

        logging.info("SVPG TS: {}, Agent TS: {}".format(
            svpg_timesteps, simulator_agent.agent_timesteps))

        solved, info = simulator_agent.select_action(agent_policy)
        svpg_timesteps += 1

        # if info is not None:
        #     new_best = stats_logger.update(args, paths, info)
        #
        #     if new_best:
        #         agent_policy.save(filename='best-seed{}'.format(args.seed), directory=paths['paper'])
        #         if args.save_particles:
        #             simulator_agent.svpg.save(directory=paths['particles'])
        #
        #         generalization_metric = visualizer.generate_ground_truth(simulator_agent, agent_policy, svpg_timesteps,
        #         log_path=paths['groundtruth_logs'])
        #
        #         np.savez('{}/best-generalization-seed{}.npz'.format(paths['paper'], args.seed),
        #             generalization_metric=generalization_metric,
        #             svpg_timesteps=svpg_timesteps,
        #             learning_curve_timesteps=simulator_agent.agent_timesteps
        #         )
        #
        #     if solved:
        #         logging.info("[SOLVED]")

    # agent_policy.save(filename='final-seed{}'.format(args.seed), directory=paths['paper'])
    # visualizer.plot_reward(simulator_agent, agent_policy,
    #         svpg_timesteps, log_path=paths['policy_logs'], plot_path=paths['policy_plots'])
    # visualizer.plot_sampling_frequency(simulator_agent, agent_policy,
    #     svpg_timesteps, log_path=paths['sampling_logs'], plot_path=paths['sampling_plots'])
    # reshow_hyperparameters(args, paths)
