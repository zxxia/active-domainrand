class MultiEnv():
    """A wrapper around multiple pensieve environments.

    The idea is to mimic RandomizedSubprocVecEnv in ADR. For now,
    multiprocessing is not necessary.
    """

    def __init__(self, envs):
        self.envs = envs
        raise NotImplementedError

    def randomize(self, randomized_values):
        # TODO: randomize each
        for env in self.envs:
            env.randomize()
        raise NotImplementedError

    def reset(self):
        for env in self.envs:
            env.reset()

    def get_current_params(self):
        # TODO: get current parameters from each individual environment
        raise NotImplementedError
