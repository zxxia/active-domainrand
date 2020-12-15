
import csv
import logging
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from pensieve.a3c import A3C, compute_entropy
from pensieve.constants import (A_DIM, ACTOR_LR_RATE, BUFFER_NORM_FACTOR,
                                CRITIC_LR_RATE, DEFAULT_QUALITY, M_IN_K,
                                S_INFO, S_LEN, TRAIN_SEQ_LEN, VIDEO_BIT_RATE)
from pensieve.utils import linear_reward


class Pensieve(object):
    def __init__(self, num_agents, log_dir, actor_path=None,
                 critic_path=None, model_save_interval=100):
        """Construct Pensieve object.

        Args
            num_agents(int): number of processes to train pensieve models.
            log_dir(str): path where all log files and model checkpoints will
            be saved to.
            actor_path(None or str): path to a actor checkpoint to be loaded.
            critic_path(None or str): path to a critic checkpoint to be loaded.
            model_save_interval(int): the period of caching model checkpoints.
        """
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

    def train(self, net_envs, iters):
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
                                       self.log_dir, self.model_save_interval))
        coordinator.start()

        agents = []
        for i in range(self.num_agents):
            agents.append(mp.Process(target=agent,
                                     args=(i, net_params_queues[i],
                                           exp_queues[i], iters, net_envs,
                                           self.log_dir)))
        for i in range(self.num_agents):
            agents[i].start()

        # wait unit training is done
        coordinator.join()
        for i in range(self.num_agents):
            agents[i].join()

        self.epoch += iters

    def test(self, net_envs):
        for net_env in net_envs:
            net_env.reset()
            time_stamp = 0
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            state = torch.zeros((S_INFO, S_LEN))
            log_path = os.path.join(
                self.summary_dir, 'log_sim_rl_' + net_env.filename)
            with open(log_path, 'w', 1) as f:
                csv_writer = csv.writer(f, delimiter='\t')
                while True:  # serve video forever
                    # the action is from the last decision
                    # this is to make the framework similar to the real
                    delay, sleep_time, buffer_size, rebuf, \
                        video_chunk_size, next_video_chunk_sizes, \
                        end_of_video, video_chunk_remain = net_env.step(
                            bit_rate)

                    time_stamp += delay  # in ms
                    time_stamp += sleep_time  # in ms

                    # reward is video quality - rebuffer penalty - smoothness
                    reward = linear_reward(bit_rate, last_bit_rate, rebuf)

                    last_bit_rate = bit_rate

                    # log time_stamp, bit_rate, buffer_size, reward
                    csv_writer.writerow([time_stamp / M_IN_K,
                                         VIDEO_BIT_RATE[bit_rate],
                                         buffer_size, rebuf, video_chunk_size,
                                         delay, reward])

                    # retrieve previous state
                    state = torch.roll(state, -1, dims=-1)

                    # this should be S_INFO number of terms
                    state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                        float(np.max(VIDEO_BIT_RATE))  # last quality
                    state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                    state[2, -1] = float(video_chunk_size) / \
                        float(delay) / M_IN_K  # kilo byte / ms
                    state[3, -1] = float(delay) / M_IN_K / \
                        BUFFER_NORM_FACTOR  # 10 sec
                    state[4, :A_DIM] = torch.tensor(
                        next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                    state[5, -1] = video_chunk_remain / \
                        net_env.total_video_chunk

                    with torch.no_grad():
                        probability = self.net.forward(state.unsqueeze(0))
                        # m = Categorical(probability)
                        # bit_rate = m.sample().item()
                        bit_rate = probability.argmax().item()
                    # Note: we need to discretize the probability into
                    # 1/RAND_RANGE steps, because there is an intrinsic
                    # discrepancy in passing single state and batch states

                    if end_of_video:
                        break

    def select_action(self, state):
        # bitrate, action_prob_vec
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).type('torch.FloatTensor')
        if not torch.is_tensor(state):
            raise TypeError
        bit_rate, _ = self.net.select_action(state)
        return bit_rate

    def save_models(self, model_save_path):
        self.net.save_actor_model(model_save_path)
        self.net.save_critic_model(model_save_path)

    def load_models(self, actor_model_path, critic_model_path):
        if actor_model_path is not None:
            self.net.load_actor_model(actor_model_path)
        if critic_model_path is not None:
            self.net.load_critic_model(critic_model_path)


def central_agent(num_agents, net, net_params_queues, exp_queues, iters,
                  summary_dir, model_save_interval):
    torch.set_num_threads(1)

    logging.basicConfig(filename=os.path.join(summary_dir, 'log_central'),
                        filemode='w', level=logging.INFO)

    assert net.is_central
    # test_log_file = open(log_file_name, 'w')
    #
    # test_log_file.write("\t".join(
    #     ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
    #      'rewards_median', 'rewards_95per', 'rewards_max\n']))
    # train_e2e_log_file = open(LOG_FILE+'_train_e2e', 'w')
    # train_e2e_log_file.write("\t".join(
    #     ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
    #      'rewards_median', 'rewards_95per', 'rewards_max\n']))
    # val_log_file = open(LOG_FILE+'_val', 'w')
    # val_log_file.write("\t".join(
    #     ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
    #      'rewards_median', 'rewards_95per', 'rewards_max\n']))

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

        if (epoch+1) % model_save_interval == 0:
            # Save the neural net parameters to disk.
            print("\nTrain epoch: {}, time use: {}s".format(
                epoch + 1, time.time() - t_start))
            t_start = time.time()
            net.save_critic_model(summary_dir)
            net.save_actor_model(summary_dir)
            # testing(epoch+1, SUMMARY_DIR+"/actor.pt",
            #         test_log_file, TEST_LOG_FOLDER, TEST_TRACES)
            # testing(epoch+1, SUMMARY_DIR+"/actor.pt",
            #         train_e2e_log_file, TEST_LOG_FOLDER, TRAIN_TRACES)
            # testing(epoch+1, SUMMARY_DIR+"/actor.pt",
            #         val_log_file, TEST_LOG_FOLDER, VAL_TRACES)

    # signal all agents to exit, otherwise they block forever.
    for i in range(num_agents):
        net_params_queues[i].put("exit")


def agent(agent_id, net_params_queue, exp_queue, iters, net_envs, summary_dir):
    torch.set_num_threads(1)
    # set random seed
    np.random.seed(agent_id)

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

            env_idx = np.random.randint(len(net_envs))
            net_env = net_envs[env_idx]
            # print('agent_{}: iter{}/{}, env_idx={}'.format(
            #     agent_id, epoch, iters, env_idx))

            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.step(bit_rate)
            state = torch.from_numpy(state).type('torch.FloatTensor')

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms

            while not end_of_video and len(s_batch) < TRAIN_SEQ_LEN:
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
