import torch
import numpy as np


class MemoryBuffer:
    """
    Episodic replay buffer for variable-length episodes.
    Stores full episodes (lists/ndarrays) and returns a padded batch.
    Collated tensors are torch.FloatTensor (CPU) — caller should .to(device).
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.episodes = []
        self.position = 0

    def push(self, episode_data: dict):
        """Store a complete episode."""
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode_data)
        else:
            self.episodes[self.position] = episode_data
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict:
        """Sample batch of episodes for training."""
        indices = np.random.choice(len(self.episodes), min(batch_size, len(self.episodes)), replace=False)
        batch = [self.episodes[i] for i in indices]
        return self._collate_episodes(batch)

    @staticmethod
    def _collate_episodes(episodes: list) -> dict:
        """Collate episodes into training batches."""
        if not episodes:
            return {}

        # Find maximum episode length for padding
        max_length = max(len(episode['observations']) for episode in episodes)

        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_masks = []

        for episode in episodes:
            obs = episode['observations']
            actions = episode['actions']
            rewards = episode['rewards']
            values = episode['values']

            # Ensure all arrays are at least 2D
            if obs.ndim == 1:
                obs = obs.reshape(-1, 1)
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            if rewards.ndim == 0:
                rewards = np.array([rewards])
            elif rewards.ndim == 1 and len(rewards) == 1:
                rewards = rewards.reshape(-1)
            if values.ndim == 0:
                values = np.array([values])
            elif values.ndim == 1 and len(values) == 1:
                values = values.reshape(-1)

            # Pad sequences to max length
            pad_length = max_length - len(obs)
            if pad_length > 0:
                # Pad observations
                obs_padded = np.concatenate([
                    obs,
                    np.zeros((pad_length, obs.shape[1]))
                ])

                # Pad actions
                actions_padded = np.concatenate([
                    actions,
                    np.zeros((pad_length, actions.shape[1]))
                ])

                # Pad rewards and values (1D arrays)
                rewards_padded = np.concatenate([
                    rewards,
                    np.zeros(pad_length)
                ])
                values_padded = np.concatenate([
                    values,
                    np.zeros(pad_length)
                ])
                mask = np.concatenate([
                    np.ones(len(obs)),
                    np.zeros(pad_length)
                ])
            else:
                obs_padded = obs
                actions_padded = actions
                rewards_padded = rewards
                values_padded = values
                mask = np.ones(len(obs))

            batch_obs.append(obs_padded)
            batch_actions.append(actions_padded)
            batch_rewards.append(rewards_padded)
            batch_values.append(values_padded)
            batch_masks.append(mask)

        return {
            'observations': torch.FloatTensor(np.array(batch_obs)),
            'actions': torch.FloatTensor(np.array(batch_actions)),
            'rewards': torch.FloatTensor(np.array(batch_rewards)),
            'values': torch.FloatTensor(np.array(batch_values)),
            'masks': torch.FloatTensor(np.array(batch_masks))
        }

    def __len__(self):
        return len(self.episodes)