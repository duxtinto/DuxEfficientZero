class Game:
    def __init__(self, env, action_space_size: int, discount: float, config=None):
        self.env = env
        self.action_space_size = action_space_size
        self.discount = discount
        self.config = config

    def legal_actions(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)