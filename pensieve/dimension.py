import numpy as np


class Dimension(object):
    """Class which handles a parameter of Pensieve environment."""

    def __init__(self, default_value, seed, min_value, max_value, name, unit):
        """Initialize Dimension object."""
        self.default_value = default_value
        self.current_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.name = name
        self.unit = unit

        # TODO: doesn't this change the random seed for all numpy uses?
        np.random.seed(seed)

    def randomize(self):
        """Uniformly randomize Dimension's value."""
        self.current_value = np.random.uniform(low=self.min_value,
                                               high=self.max_value)

    def reset(self):
        """Reset Dimension's value to default."""
        self.current_value = self.default_value

    def set(self, value):
        """Set Dimension's value."""
        self.current_value = value
