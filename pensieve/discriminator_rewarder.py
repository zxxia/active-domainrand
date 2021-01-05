import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from common.models.discriminator import MLPDiscriminator
from pensieve.agent_policy.pensieve import Pensieve, RobustMPC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SVPGReward(object):
    def __init__(self, nagents, nparams, svpg_rollout_length):
        self.nagents = nagents
        self.nparams = nparams
        self.svpg_rollout_length = svpg_rollout_length

    def calculate_rewards(self, test_envs, last_ref_reward):
        '''
        Calculate RL-MPC reward for SVPG to learn
        :param test_envs: all the particles of current env parameter setting
        :param last_ref_reward: RL-MPC reward for all particles from the last round of test_envs
        :return: RL-MPC reward for all particles
        '''

        ref_reward_all = np.zeros((self.nagents, self.svpg_rollout_length, 1))

        for i in range(self.nagents):
            for t in range(self.svpg_rollout_length):

                # RL reward
                rl_results = Pensieve.evaluate_envs( test_envs[i][t] )
                rl_vid_rewards = [np.array( vid_results )[1: ,-1]
                               for vid_results in rl_results]
                RL_avg_chunk_reward = np.mean( np.concatenate( rl_vid_rewards ) )

                # MPC reward
                mpc_results = RobustMPC.evaluate_envs( test_envs[i][t] )
                mpc_vid_rewards = [np.array( vid_results )[1: ,-1]
                               for vid_results in mpc_results]
                MPC_avg_chunk_reward = np.mean( np.concatenate( mpc_vid_rewards ) )

                ref_reward = RL_avg_chunk_reward - MPC_avg_chunk_reward
                if ref_reward < 0 or ref_reward > 0 & last_ref_reward[i][t]<0:
                    reward_weight = 100
                else:
                    reward_weight = 1
                ref_reward = reward_weight * ref_reward

                ref_reward_all[i][t] = ref_reward

        return ref_reward_all


class DiscriminatorRewarder(object):
    def __init__(self, state_dim, action_dim, discriminator_batchsz, reward_scale,
                 load_discriminator, discriminator_lr=3e-3, add_pz=True):
        self.discriminator = MLPDiscriminator(
            state_dim=state_dim, action_dim=action_dim).to(device)

        self.discriminator_criterion = nn.BCELoss()
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr)
        self.reward_scale = reward_scale
        self.batch_size = discriminator_batchsz
        self.add_pz = add_pz

        # if load_discriminator:
        #     self._load_discriminator(randomized_env_id)

    def calculate_rewards(self, randomized_trajectory):
        """Discriminator based reward calculation
        We want to use the negative of the adversarial calculation (Normally,
        -log(D)). We want to *reward* our simulator for making it easier to
        discriminate between the reference env + randomized onea
        """
        score, _, _ = self.get_score(randomized_trajectory)
        reward = np.log(score)

        if self.add_pz:
            reward -= np.log(0.5)

        return self.reward_scale * reward

    def get_score(self, trajectory):
        """Discriminator based reward calculation
        We want to use the negative of the adversarial calculation (Normally,
        -log(D)). We want to *reward* our simulator for making it easier to
        discriminate between the reference env + randomized onea
        """
        traj_tensor = self._trajectory2tensor(trajectory).float()

        with torch.no_grad():
            score = (self.discriminator(
                traj_tensor).cpu().detach().numpy()+1e-8)
            return score.mean(), np.median(score), np.sum(score)

    def train_discriminator(self, reference_trajectory, randomized_trajectory, iterations):
        """Trains discriminator to distinguish between reference and randomized state action tuples
        """
        for _ in range(iterations):
            randind = np.random.randint(
                0, len(randomized_trajectory[0]), size=int(self.batch_size))
            refind = np.random.randint(
                0, len(reference_trajectory[0]), size=int(self.batch_size))

            randomized_batch = self._trajectory2tensor(
                randomized_trajectory[randind])
            reference_batch = self._trajectory2tensor(
                reference_trajectory[refind])

            g_o = self.discriminator(randomized_batch)
            e_o = self.discriminator(reference_batch)

            self.discriminator_optimizer.zero_grad()

            discrim_loss = self.discriminator_criterion(g_o, torch.ones((len(randomized_batch), 1), device=device)) + \
                self.discriminator_criterion(e_o, torch.zeros(
                    (len(reference_batch), 1), device=device))
            discrim_loss.backward()

            self.discriminator_optimizer.step()

    def _load_discriminator(self, name, path='saved-models/discriminator/discriminator_{}.pth'):
        self.discriminator.load_state_dict(
            torch.load(path.format(name), map_location=device))

    def _save_discriminator(self, name, path='saved-models/discriminator/discriminator_{}.pth'):
        torch.save(self.discriminator.state_dict(), path.format(name))

    def _trajectory2tensor(self, trajectory):
        return torch.from_numpy(trajectory).float().to(device)
