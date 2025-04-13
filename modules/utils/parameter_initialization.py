import numpy as np

class ParametersInit:
    """Training parameters initialization."""

    def __init__(self, config):
        self.config = config

        # Initialization function according to the init_type
        self.init_function = getattr(self, self.config.init_type.name)

    def __call__(self, layer):
        """Initializing layer parameters"""
        for param_name in layer.parameters:
            param = getattr(layer, param_name)

            if param_name == 'bias' and self.config.zero_bias:
                setattr(layer, param_name, self.zeros(param.shape))
            else:
                setattr(layer, param_name, self.init_function(param.shape))

    def zeros(self, param_shape: tuple) -> np.ndarray:
        """Initialization with zeros."""
        return np.zeros(param_shape)

    def normal(self, param_shape: tuple) -> np.ndarray:
        """Initialization with values from a normal distribution.

        W ~ N(mu, sigma^2),

        where:
            - mu and sigma can be defined in self.config.init_kwargs
        """
        return np.random.normal(loc=self.config.init_kwargs['mu'], scale=self.config.init_kwargs['sigma'], size=param_shape)

    def uniform(self, param_shape: tuple) -> np.ndarray:
        """Initialization with values from a uniform distribution.

        W ~ U(-epsilon, epsilon),

        where:
            - epsilon can be defined in self.config.init_kwargs
        """
        return np.random.uniform(low=-self.config.init_kwargs['epsilon'], high=self.config.init_kwargs['epsilon'], size=param_shape)

    def he(self, param_shape: tuple) -> np.ndarray:
        """He initialization.
        
        Initialization with values from a normal distribution with the following parameters:

        W ~ N(0, 2 / M_{l-1}),

        where:
            - M_{l-1} - number of input features of a layer
        """
        return np.random.normal(loc=0, scale=np.sqrt(2 / param_shape[1]), size=param_shape)

    def xavier(self, param_shape: tuple) -> np.ndarray:
        """Xavier initialization.
        
        Initialization with values from a uniform distribution with the following parameters:

        W ~ U(-epsilon, epsilon),

        where:
            - epsilon = \sqrt{1 / M_{l-1}},
            - M_{l-1} - number of input features of a layer
        """
        epsilon = np.sqrt(1 / param_shape[1])
        return np.random.uniform(low=-epsilon, high=epsilon, size=param_shape)

    def xavier_normalized(self, param_shape: tuple) -> np.ndarray:
        """Xavier normalized initialization.
        
        Initialization with values from a uniform distribution with the following parameters:

        W ~ U(-epsilon, epsilon),

        where:
            - epsilon = \sqrt{6 / (M_{l-1} + M_l)},
            - M_{l-1} - number of input features of a layer,
            - M_l - number of output features of a layer.
        """
        epsilon = np.sqrt(6.0 / sum(param_shape))
        return np.random.uniform(low=-epsilon, high=epsilon, size=param_shape)
