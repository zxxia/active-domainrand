
import csv
import logging
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from pensieve.a3c import A3C, compute_entropy
from pensieve.agent_policy import BaseAgentPolicy
from pensieve.constants import (A_DIM, ACTOR_LR_RATE, CRITIC_LR_RATE,
                                DEFAULT_QUALITY, M_IN_K, S_INFO, S_LEN,
                                VIDEO_BIT_RATE)


class Pensieve(BaseAgentPolicy):
    def __init__(self, num_agents, log_dir, actor_path=None,
                 critic_path=None, model_save_interval=100, batch_size=100):
        """Construct Pensieve object.

        Args
            num_agents(int): number of processes to train pensieve models.
            log_dir(str): path where all log files and model checkpoints will
            be saved to.
            actor_path(None or str): path to a actor checkpoint to be loaded.
            critic_path(None or str): path to a critic checkpoint to be loaded.
            model_save_interval(int): the period of caching model checkpoints.
            batch_size(int): training batch size.
        """
        # https://github.com/pytorch/pytorch/issues/3966
        mp.set_start_method("spawn")
        self.num_agents = num_agents

        self.net = A3C(True, [S_INFO, S_LEN], A_DIM,
                       ACTOR_LR_RATE, CRITIC_LR_RATE)
        # NOTE: this is required for the ``fork`` method to work
        self.net.actor_network.share_memory()
        self.net.critic_network.share_memory()

        self.load_models(actor_path, critic_path)

        self.log_dir = log_dir
        self.model_save_interval = model_save_interval
        self.epoch = 0  # track how many epochs the models have been trained
        self.batch_size = batch_size

    def train(self, train_envs, val_envs=None, test_envs=None, iters=1e5,
              reference_agent_policy=None):
        for net_env in train_envs:
            net_env.reset()
        # inter-process communication queues
        net_params_queues = []
        exp_queues = []
        for i in range(self.num_agents):
            net_params_queues.append(mp.Queue(1))
            exp_queues.append(mp.Queue(1))

        # create a coordinator and multiple agent processes
        # (note: threading is not desirable due to python GIL)
        assert len(net_params_queues) == self.num_agents
        assert len(exp_queues) == self.num_agents
        coordinator = mp.Process(target=central_agent,
                                 args=(self.num_agents, self.net,
                                       net_params_queues, exp_queues, iters,
                                       self.log_dir, self.model_save_interval,
                                       val_envs, self.epoch))
        coordinator.start()

        agents = []
        for i in range(self.num_agents):
            agents.append(mp.Process(target=agent,
                                     args=(i, net_params_queues[i],
                                           exp_queues[i], train_envs,
                                           self.log_dir, self.batch_size)))
        for i in range(self.num_agents):
            agents[i].start()

        # wait unit training is done
        coordinator.join()
        for i in range(self.num_agents):
            agents[i].join()

        self.epoch += iters

    def evaluate(self, net_env):
        return evaluate_env(self.net, net_env)

    def test(self, net_envs):
        for net_env in net_envs:
            self.evaluate(net_env)

            raise NotImplementedError
        # TODO: return trajectories consider multiprocessing

    def select_action(self, state):
        raise NotImplementedError

    def save_models(self, model_save_path):
        """Save models to a directory."""
        self.net.save_actor_model(os.path.join(model_save_path, "actor.pth"))
        self.net.save_critic_model(os.path.join(model_save_path, "critic.pth"))

    def load_models(self, actor_model_path, critic_model_path):
        """Load models from given paths."""
        if actor_model_path is not None:
            self.net.load_actor_model(actor_model_path)
        if critic_model_path is not None:
            self.net.load_critic_model(critic_model_path)


def central_agent(num_agents, net, net_params_queues, exp_queues, iters,
                  summary_dir, model_save_interval, val_envs, epoch_trained):
    torch.set_num_threads(1)

    logging.basicConfig(filename=os.path.join(summary_dir, 'log_central'),
                        filemode='w', level=logging.INFO)

    assert net.is_central
    test_log_writer = csv.writer(
        open(os.path.join(summary_dir, 'log_test'), 'w', 1), delimiter='\t')
    test_log_writer.writerow(
        ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
         'rewards_median', 'rewards_95per', 'rewards_max'])

    train_e2e_log_writer = csv.writer(
        open(os.path.join(summary_dir, 'log_train_e2e'), 'w', 1),
        delimiter='\t')
    train_e2e_log_writer.writerow(
        ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
         'rewards_median', 'rewards_95per', 'rewards_max'])

    val_log_writer = csv.writer(
        open(os.path.join(summary_dir, 'log_val'), 'w', 1), delimiter='\t')
    val_log_writer.writerow(
        ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
         'rewards_median', 'rewards_95per', 'rewards_max'])

    t_start = time.time()
    for epoch in range(iters):
        # synchronize the network parameters of work agent
        actor_net_params = net.get_actor_param()
        # critic_net_params=net.getCriticParam()
        for i in range(num_agents):
            # net_params_queues[i].put([actor_net_params,critic_net_params])
            net_params_queues[i].put(actor_net_params)
            # Note: this is synchronous version of the parallel training,
            # which is easier to understand and probe. The framework can be
            # fairly easily modified to support asynchronous training.
            # Some practices of asynchronous training (lock-free SGD at
            # its core) are nicely explained in the following two papers:
            # https://arxiv.org/abs/1602.01783
            # https://arxiv.org/abs/1106.5730

        # record average reward and td loss change
        # in the experiences from the agents
        total_batch_len = 0.0
        total_reward = 0.0
        # total_td_loss = 0.0
        total_entropy = 0.0
        total_agents = 0.0

        # assemble experiences from the agents
        # actor_gradient_batch = []
        # critic_gradient_batch = []

        for i in range(num_agents):
            # print('central_agent: {}/{}'.format(epoch, iters))
            s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
            net.get_network_gradient(
                s_batch, a_batch, r_batch, terminal=terminal, epoch=epoch)

            total_reward += np.sum(r_batch)
            total_batch_len += len(r_batch)
            total_agents += 1.0
            total_entropy += np.sum(info['entropy'])

        # log training information
        net.update_network()

        avg_reward = total_reward / total_agents
        avg_entropy = total_entropy / total_batch_len

        logging.info('Epoch: {} Avg_reward: {} Avg_entropy: {}'.format(
            epoch, avg_reward, avg_entropy))

        if (epoch_trained+epoch+1) % model_save_interval == 0:
            # Save the neural net parameters to disk.
            print("\nTrain epoch: {}/{}, time use: {}s".format(
                epoch + 1, iters, time.time() - t_start))
            t_start = time.time()
            net.save_critic_model(os.path.join(
                summary_dir, "critic_ep_{}".format(epoch_trained+epoch+1)))
            net.save_actor_model(os.path.join(
                summary_dir, "actor_ep_{}".format(epoch_trained+epoch+1)))
            val_results = evaluate_envs(net, val_envs)
            # TODO: process val results and write into log
            # evaluate_envs(net, train_envs)
            # evaluate_envs(net, test_envs)

    # signal all agents to exit, otherwise they block forever.
    for i in range(num_agents):
        net_params_queues[i].put("exit")


def evaluate_env(net, net_env):
    net_env.reset()
    results = []
    time_stamp = 0
    bit_rate = DEFAULT_QUALITY
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        state, reward, end_of_video, info = net_env.step(bit_rate)

        time_stamp += info['delay']  # in ms
        time_stamp += info['sleep_time']  # in ms

        results.append([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                        info['buffer_size'], info['rebuf'],
                        info['video_chunk_size'], info['delay'], reward])

        state = torch.from_numpy(state).type('torch.FloatTensor')
        bit_rate, action_prob_vec = net.select_action(state)
        if end_of_video:
            break
    return results


def evaluate_envs(net, net_envs):
    results = []
    for net_env in net_envs:
        results.append(evaluate_env(net, net_env))
    return results


def agent(agent_id, net_params_queue, exp_queue, net_envs, summary_dir,
          batch_size):
    torch.set_num_threads(1)
    # set random seed
    prng = np.random.RandomState(agent_id)

    with open(os.path.join(summary_dir,
                           'log_agent_'+str(agent_id)), 'w', 1) as log_file:
        csv_writer = csv.writer(log_file, delimiter='\t')

        csv_writer.writerow(['time_stamp', 'bit_rate', 'buffer_size',
                             'rebuffer', 'video_chunk_size', 'delay',
                             'reward',
                             'epoch', 'trace_idx', 'mahimahi_ptr'])

        net = A3C(False, [S_INFO, S_LEN], A_DIM, ACTOR_LR_RATE, CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the
        # coordinator

        time_stamp = 0
        epoch = 0
        while True:
            actor_net_params = net_params_queue.get()
            if actor_net_params == "exit":
                break
            net.hard_update_actor_network(actor_net_params)
            bit_rate = DEFAULT_QUALITY
            s_batch = []
            a_batch = []
            r_batch = []
            entropy_record = []
            # state = torch.zeros((1, S_INFO, S_LEN))

            env_idx = prng.randint(len(net_envs))
            net_env = net_envs[env_idx]
            # print('agent_{}: iter{}/{}, env_idx={}'.format(
            #     agent_id, epoch, iters, env_idx))

            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.step(bit_rate)
            state = torch.from_numpy(state).type('torch.FloatTensor')

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms

            while not end_of_video and len(s_batch) < batch_size:
                bit_rate, action_prob_vec = net.select_action(state)
                # Note: we need to discretize the probability into 1/RAND_RANGE
                # steps, because there is an intrinsic discrepancy in passing
                # single state and batch states

                state, reward, end_of_video, info = net_env.step(bit_rate)
                state = torch.from_numpy(state).type('torch.FloatTensor')

                s_batch.append(state)
                a_batch.append(bit_rate)
                r_batch.append(reward)
                entropy_record.append(compute_entropy(action_prob_vec))

                # log time_stamp, bit_rate, buffer_size, reward
                csv_writer.writerow([time_stamp, VIDEO_BIT_RATE[bit_rate],
                                     info['buffer_size'], info['rebuf'],
                                     info['video_chunk_size'], info['delay'],
                                     reward, epoch, env_idx])
            # print('agent_{} put {}'.format(agent_id, len(s_batch)))
            exp_queue.put([s_batch,  # ignore the first chuck
                           a_batch,  # since we don't have the
                           r_batch,  # control over it
                           end_of_video,
                           {'entropy': entropy_record}])
            if end_of_video:
                net_env.reset()

            log_file.write('\n')  # so that in the log we know where video ends
            epoch += 1
