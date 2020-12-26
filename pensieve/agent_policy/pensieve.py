
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
        # mp.set_start_method("spawn")
        self.num_agents = num_agents

        self.net = A3C(True, [S_INFO, S_LEN], A_DIM,
                       ACTOR_LR_RATE, CRITIC_LR_RATE)
        # NOTE: this is required for the ``fork`` method to work
        # self.net.actor_network.share_memory()
        # self.net.critic_network.share_memory()

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

        agents = []
        for i in range(self.num_agents):
            agents.append(mp.Process(target=agent,
                                     args=(i, net_params_queues[i],
                                           exp_queues[i], train_envs,
                                           self.log_dir, self.batch_size)))
        for i in range(self.num_agents):
            agents[i].start()

        self.central_agent(net_params_queues, exp_queues, iters, val_envs,
                           test_envs)

        # wait unit training is done
        for i in range(self.num_agents):
            agents[i].join()

        self.epoch += iters

    def evaluate(self, net_env):
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
            bit_rate, action_prob_vec = self.net.select_action(state)
            bit_rate = np.argmax(action_prob_vec)
            if end_of_video:
                break
        return results

    def evaluate_envs(self, net_envs):
        # TODO: return trajectories consider multiprocessing
        results = []
        for net_env in net_envs:
            results.append(self.evaluate(net_env))
        return results

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

    def central_agent(self, net_params_queues, exp_queues, iters, val_envs,
                      test_envs):
        torch.set_num_threads(2)

        logging.basicConfig(filename=os.path.join(self.log_dir, 'log_central'),
                            filemode='w', level=logging.INFO)

        assert self.net.is_central
        test_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_test'), 'w', 1),
            delimiter='\t')
        test_log_writer.writerow(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max'])

        train_e2e_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_train_e2e'), 'w', 1),
            delimiter='\t')
        train_e2e_log_writer.writerow(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max'])

        val_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_val'), 'w', 1),
            delimiter='\t')
        val_log_writer.writerow(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max'])

        t_start = time.time()
        for epoch in range(int(iters)):
            # synchronize the network parameters of work agent
            actor_net_params = self.net.get_actor_param()
            actor_net_params = [params.detach().cpu().numpy()
                                for params in actor_net_params]

            for i in range(self.num_agents):
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
            for i in range(self.num_agents):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
                self.net.get_network_gradient(
                    s_batch, a_batch, r_batch, terminal=terminal, epoch=epoch)
                total_reward += np.sum(r_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])
            print('central_agent: {}/{}'.format(epoch, iters))

            # log training information
            self.net.update_network()

            avg_reward = total_reward / total_agents
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: {} Avg_reward: {} Avg_entropy: {}'.format(
                epoch, avg_reward, avg_entropy))

            if (self.epoch+epoch+1) % self.model_save_interval == 0:
                # Save the neural net parameters to disk.
                print("Train epoch: {}/{}, time use: {}s".format(
                    epoch + 1, iters, time.time() - t_start))
                self.net.save_critic_model(os.path.join(
                    self.log_dir, "critic_ep_{}.pth".format(self.epoch+epoch+1)))
                self.net.save_actor_model(os.path.join(
                    self.log_dir, "actor_ep_{}.pth".format(self.epoch+epoch+1)))
                if val_envs is not None:
                    val_results = self.evaluate_envs(val_envs)
                    vid_rewards = [np.sum(np.array(vid_results)[1:, -1])
                                   for vid_results in val_results]
                    val_log_writer.writerow([self.epoch + epoch + 1,
                                             np.min(vid_rewards),
                                             np.percentile(vid_rewards, 5),
                                             np.mean(vid_rewards),
                                             np.median(vid_rewards),
                                             np.percentile(vid_rewards, 95),
                                             np.max(vid_rewards)])
                if test_envs is not None:
                    test_results = self.evaluate_envs(test_envs)
                    vid_rewards = [np.sum(np.array(vid_results)[1:, -1])
                                   for vid_results in test_results]
                    test_log_writer.writerow([self.epoch + epoch + 1,
                                              np.min(vid_rewards),
                                              np.percentile(vid_rewards, 5),
                                              np.mean(vid_rewards),
                                              np.median(vid_rewards),
                                              np.percentile(vid_rewards, 95),
                                              np.max(vid_rewards)])
                t_start = time.time()
                # TODO: process val results and write into log
                # evaluate_envs(net, train_envs)

        # signal all agents to exit, otherwise they block forever.
        for i in range(self.num_agents):
            net_params_queues[i].put("exit")


def agent(agent_id, net_params_queue, exp_queue, net_envs, summary_dir,
          batch_size):
    torch.set_num_threads(1)
    # set random seed
    prng = np.random.RandomState(agent_id)

    with open(os.path.join(summary_dir,
                           'log_agent_'+str(agent_id)), 'w', 1) as log_file:
        csv_writer = csv.writer(log_file, delimiter='\t', lineterminator="\n")

        csv_writer.writerow(['time_stamp', 'bit_rate', 'buffer_size',
                             'rebuffer', 'video_chunk_size', 'delay',
                             'reward',
                             'epoch', 'trace_idx', 'mahimahi_ptr'])

        # initial synchronization of the network parameters from the
        # coordinator
        net = A3C(False, [S_INFO, S_LEN], A_DIM, ACTOR_LR_RATE, CRITIC_LR_RATE)
        actor_net_params = net_params_queue.get()
        if actor_net_params == "exit":
            return
        net.hard_update_actor_network(actor_net_params)

        time_stamp = 0
        epoch = 0
        env_idx = prng.randint(len(net_envs))
        net_env = net_envs[env_idx]
        bit_rate = DEFAULT_QUALITY
        s_batch = []
        a_batch = []
        r_batch = []
        entropy_record = []
        is_1st_step = True
        while True:

            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.step(bit_rate)

            bit_rate, action_prob_vec = net.select_action(state)
            bit_rate = bit_rate[0]
            # Note: we need to discretize the probability into 1/RAND_RANGE
            # steps, because there is an intrinsic discrepancy in passing
            # single state and batch states

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms
            if not is_1st_step:
                s_batch.append(state)
                a_batch.append(bit_rate)
                r_batch.append(reward)
                entropy_record.append(compute_entropy(action_prob_vec)[0])
            else:
                # ignore the first chunck since we can't control it
                is_1st_step = False

            # log time_stamp, bit_rate, buffer_size, reward
            csv_writer.writerow([time_stamp, VIDEO_BIT_RATE[bit_rate],
                                 info['buffer_size'], info['rebuf'],
                                 info['video_chunk_size'], info['delay'],
                                 reward, epoch, env_idx])
            if len(s_batch) == batch_size:
                exp_queue.put([np.concatenate(s_batch), np.array(a_batch),
                               np.array(r_batch), end_of_video,
                               {'entropy': np.array(entropy_record)}])

                actor_net_params = net_params_queue.get()
                if actor_net_params == "exit":
                    break
                net.hard_update_actor_network(actor_net_params)
                s_batch = []
                a_batch = []
                r_batch = []
                entropy_record = []
                epoch += 1
            if end_of_video:
                net_env.reset()
                env_idx = prng.randint(len(net_envs))
                net_env = net_envs[env_idx]
                bit_rate = DEFAULT_QUALITY
                is_1st_step = True
                log_file.write('\n')  # mark video ends in log
