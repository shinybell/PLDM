"""
Wrappers for MiniGrid environments to support PLDM-compatible observations.

These wrappers transform MiniGrid observations into the format required by PLDM,
including resizing to 64x64 and proper channel ordering.
"""

import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium import spaces
import copy


class GoalRenderingMixin:
    """
    Mixin that adds goal rendering capability to MiniGrid wrappers.

    This allows rendering what the environment would look like if the agent
    were at the goal position, similar to DiverseMaze's get_target_obs().
    """

    def render_at_position(self, position, direction=None):
        """
        Render the environment with the agent at a specific position.

        Args:
            position: (x, y) tuple/array of grid position
            direction: Agent direction (0-3). If None, uses current direction.

        Returns:
            RGB image array of shape (H, W, 3)
        """
        # Get unwrapped environment
        unwrapped = self.unwrapped

        # Save current agent state
        original_pos = unwrapped.agent_pos.copy()
        original_dir = unwrapped.agent_dir

        # Set agent to target position
        unwrapped.agent_pos = np.array(position, dtype=np.int32)
        if direction is not None:
            unwrapped.agent_dir = direction

        # Render at this position
        rgb_image = unwrapped.render()

        # Restore original agent state
        unwrapped.agent_pos = original_pos
        unwrapped.agent_dir = original_dir

        return rgb_image

    def get_target_obs(self, target_position=None, direction=None):
        """
        Get observation at the target/goal position.

        Args:
            target_position: (x, y) tuple/array. If None, uses env's goal_pos.
            direction: Agent direction. If None, uses current direction.

        Returns:
            Observation (RGB image) at the target position, processed through
            all observation wrappers.
        """
        # Get unwrapped environment
        unwrapped = self.unwrapped

        # Use environment's goal position if not specified
        if target_position is None:
            if hasattr(unwrapped, 'goal_pos'):
                target_position = unwrapped.goal_pos
            else:
                raise ValueError("Environment doesn't have goal_pos and target_position not provided")

        # Render at target position
        rgb_image = self.render_at_position(target_position, direction)

        # Apply observation transformations (resize, etc.)
        # This processes the raw RGB through the observation wrappers
        return self.observation(rgb_image)


class ResizeObservationWrapper(gym.ObservationWrapper, GoalRenderingMixin):
    """
    Wrapper that resizes RGB observations to a specified size.

    This wrapper is designed for PLDM compatibility, which expects 64x64 RGB images.
    It takes the rendered RGB image from MiniGrid and resizes it to the target size.

    Args:
        env: The MiniGrid environment to wrap
        size: Tuple of (height, width) for the resized observation (default: (64, 64))
        render_mode: Render mode to use for getting RGB images (default: 'rgb_array')

    Example:
        env = gym.make('MiniGrid-LongHorizon-Level1-v0')
        env = ResizeObservationWrapper(env, size=(64, 64))
        obs, info = env.reset()
        # obs.shape will be (64, 64, 3)
    """

    def __init__(self, env, size=(64, 64), render_mode='rgb_array'):
        super().__init__(env)

        # size can be int or tuple
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.height, self.width = self.size

        # Update observation space to match resized image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8
        )

    def observation(self, observation):
        """
        Transform the observation by resizing.

        Args:
            observation: RGB image observation from the environment
                        Can be either:
                        - numpy array of shape (H, W, 3)
                        - dict with 'image' key containing the RGB array

        Returns:
            Resized RGB image of shape (height, width, 3)
        """
        # Extract image from observation (handle both dict and array cases)
        if isinstance(observation, dict):
            rgb_image = observation['image']
        else:
            rgb_image = observation

        # Resize using PIL (high quality)
        pil_image = Image.fromarray(rgb_image)
        resized_pil = pil_image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        resized_image = np.array(resized_pil)

        return resized_image


class ChannelFirstWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts observations from (H, W, C) to (C, H, W) format.

    This is useful for PyTorch models which expect channel-first format.

    Args:
        env: The environment to wrap

    Example:
        env = gym.make('MiniGrid-LongHorizon-Level1-v0')
        env = ResizeObservationWrapper(env, size=(64, 64))
        env = ChannelFirstWrapper(env)
        obs, info = env.reset()
        # obs.shape will be (3, 64, 64)
    """

    def __init__(self, env):
        super().__init__(env)

        # Get original observation space
        old_shape = env.observation_space.shape

        # Create new observation space with channels first
        if len(old_shape) == 3:
            # Assume last dimension is channels
            new_shape = (old_shape[2], old_shape[0], old_shape[1])
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=new_shape,
                dtype=env.observation_space.dtype
            )
        else:
            # If not 3D, keep as is
            self.observation_space = env.observation_space

    def observation(self, observation):
        """
        Transform observation from (H, W, C) to (C, H, W).

        Args:
            observation: Observation in (H, W, C) format

        Returns:
            Observation in (C, H, W) format
        """
        if len(observation.shape) == 3:
            # Transpose from (H, W, C) to (C, H, W)
            return np.transpose(observation, (2, 0, 1))
        else:
            return observation


class NormalizeWrapper(gym.ObservationWrapper):
    """
    Wrapper that normalizes observations to [0, 1] range.

    Args:
        env: The environment to wrap

    Example:
        env = gym.make('MiniGrid-LongHorizon-Level1-v0')
        env = ResizeObservationWrapper(env, size=(64, 64))
        env = NormalizeWrapper(env)
        obs, info = env.reset()
        # obs values will be in [0, 1]
    """

    def __init__(self, env):
        super().__init__(env)

        # Update observation space to reflect normalized values
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        """
        Normalize observation from [0, 255] to [0, 1].

        Args:
            observation: Observation in [0, 255] range

        Returns:
            Observation normalized to [0, 1]
        """
        return observation.astype(np.float32) / 255.0


class PLDMWrapper(gym.Wrapper):
    """
    Composite wrapper that applies all transformations needed for PLDM.

    This wrapper:
    1. Resizes observations to 64x64
    2. Optionally converts to channel-first format (C, H, W)
    3. Optionally normalizes to [0, 1]

    Args:
        env: The MiniGrid environment to wrap
        size: Tuple of (height, width) for resized observation (default: (64, 64))
        channel_first: Whether to convert to (C, H, W) format (default: False)
        normalize: Whether to normalize to [0, 1] (default: False)

    Example:
        # For PyTorch with normalized values
        env = gym.make('MiniGrid-LongHorizon-Level1-v0')
        env = PLDMWrapper(env, channel_first=True, normalize=True)
        obs, info = env.reset()
        # obs.shape = (3, 64, 64), values in [0, 1]

        # For basic 64x64 uint8 images
        env = PLDMWrapper(env, channel_first=False, normalize=False)
        obs, info = env.reset()
        # obs.shape = (64, 64, 3), values in [0, 255]
    """

    def __init__(self, env, size=(64, 64), channel_first=False, normalize=False):
        # Apply resize wrapper
        env = ResizeObservationWrapper(env, size=size)

        # Optionally apply channel-first wrapper
        if channel_first:
            env = ChannelFirstWrapper(env)

        # Optionally apply normalization wrapper
        if normalize:
            env = NormalizeWrapper(env)

        super().__init__(env)


def make_pldm_env(env_id, size=(64, 64), channel_first=False, normalize=False, **kwargs):
    """
    Convenience function to create a PLDM-compatible MiniGrid environment.

    Args:
        env_id: Environment ID (e.g., 'MiniGrid-LongHorizon-Level1-v0')
        size: Tuple of (height, width) for resized observation (default: (64, 64))
        channel_first: Whether to use (C, H, W) format (default: False)
        normalize: Whether to normalize to [0, 1] (default: False)
        **kwargs: Additional arguments to pass to gym.make()

    Returns:
        Wrapped environment ready for PLDM

    Example:
        # Create environment with default settings (64x64, HWC format, uint8)
        env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0')

        # Create environment for PyTorch (64x64, CHW format, float32 normalized)
        env = make_pldm_env(
            'MiniGrid-LongHorizon-Level1-v0',
            channel_first=True,
            normalize=True
        )
    """
    import pldm_envs.minigrid  # Ensure environments are registered

    env = gym.make(env_id, render_mode='rgb_array', **kwargs)
    env = PLDMWrapper(env, size=size, channel_first=channel_first, normalize=normalize)

    return env
