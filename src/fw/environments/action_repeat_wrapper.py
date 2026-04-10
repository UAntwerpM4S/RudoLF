import gymnasium as gym

class ActionRepeatWrapper(gym.Wrapper):
    """
    Wrapper to repeat the same action for a given number of steps.
    """
    def __init__(self, env, repeat=4):
        """Initialize the wrapper with the base environment and repeat count."""
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        """Repeat the action for up to `repeat` steps and accumulate reward."""
        total_reward = 0.0
        done = False
        truncated = False
        info = {}

        for _ in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
                
        return obs, total_reward, done, truncated, info
