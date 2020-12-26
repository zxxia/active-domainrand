import logging
from multiprocessing import Pipe, Process

import numpy as np

logger = logging.getLogger(__name__)


class MultiEnv:
    """A wrapper around multiple pensieve environments.

    The idea is to mimic RandomizedSubprocVecEnv in ADR. For now,
    multiprocessing is not necessary.
    """

    def __init__(self, nagents, envs):
        self.envs = envs
        self.closed = False
        self.nagents = nagents
        self.remotes, self.work_remotes = zip(*[Pipe()
                                                for _ in range(nagents)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env))
                   for (work_remote, remote, env)
                   in zip(self.work_remotes, self.remotes, envs)]
        for p in self.ps:
            # if the main process crashes, kill all daemonic child processes
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # self.remotes[0].send(('get_spaces', None))
        # observation_space, action_space, randomization_space =
        # self.remotes[0].recv()
        # self.randomization_space = randomization_space
        # self.viewer = None
        # VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after " \
            "calling close()"

    def randomize(self, randomized_values):
        # TODO: randomize each
        self._assert_not_closed()
        assert len(randomized_values) == len(self.remotes), "Number of " \
        "randomized_values is not equal to number of agents!"

        logger.debug('[randomize] => SENDING')
        for remote, val in zip(self.remotes, randomized_values):
            remote.send(('randomize', val))
        for remote in self.remotes:
            remote.recv()  # clear the pipe
        logger.debug('[randomize] => SENT')
        self.waiting = False

    def reset(self):
        self._assert_not_closed()
        logger.debug('[reset] => SENDING')
        for remote in self.remotes:
            remote.send(('reset', None))
        result = [remote.recv() for remote in self.remotes]
        logger.debug('[reset] => SENT')
        return np.stack(result)

    def get_current_params(self):
        logger.debug('[get_current_values] => SENDING')
        for remote in self.remotes:
            remote.send(('get_current_values', None))
        result = [remote.recv() for remote in self.remotes]
        logger.debug('[get_current_values] => SENT')
        return np.stack(result)

    def get_current_randomization_params(self):
        logger.debug('[get_current_randomization_values] => SENDING')
        for remote in self.remotes:
            remote.send(('get_current_randomization_values', None))
        result = [remote.recv() for remote in self.remotes]
        logger.debug('[get_current_randomization_values] => SENT')
        return np.stack(result)

    def step(self, actions):
        assert len(actions) == len(self.remotes), "Number of actions is not " \
            "equal to number of agents!"
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self._assert_not_closed()
        logger.debug('[step] => SENDING')
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        logger.debug('[step] => SENT')
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        logger.debug('[step] => WAITING')
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        logger.debug('[step] => DONE')
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs).squeeze(), np.stack(rews), np.stack(dones), infos

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


def worker(remote, parent_remote, env):
    parent_remote.close()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                raise NotImplementedError
                # remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                raise NotImplementedError
                # remote.send((env.observation_space, env.action_space,
                #              env.unwrapped.randomization_space))
            elif cmd == 'get_dimension_name':
                raise NotImplementedError
                # remote.send(env.unwrapped.dimensions[data].name)
            elif cmd == 'rescale_dimension':
                raise NotImplementedError
                # dimension = data[0]
                # array = data[1]
                # rescaled =env.unwrapped.dimensions[dimension]._rescale(array)
                # remote.send(rescaled)
            elif cmd == 'randomize':
                randomized_val = data
                env.randomize(randomized_val)
                remote.send(None)
            elif cmd == 'get_current_values':
                values = {}
                for dim_name, dim in env.dimensions.items():
                    values[dim_name] = dim.current_value

                remote.send(values)
            elif cmd == 'get_current_randomization_values':
                values = {}
                for dim_name, dim in env.dimensions.items():
                    if dim.max_value != dim.min_value:
                        values[dim_name] = dim.current_value
                remote.send(values)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()
