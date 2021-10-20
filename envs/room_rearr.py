import numpy as np
import gym
import torch

import sys
sys.path.append("/Users/moritzstephan/Projects/metabenchmark/")

from room_rearrangement import room_rearr_mult_env
import meta_exploration

class InstructionWrapper(meta_exploration.InstructionWrapper):
  """InstructionWrapper for RoomRearrangeEnv."""

  def __init__(self, env, exploration_trajectory, **kwargs):
    super().__init__(env, exploration_trajectory, **kwargs)
    self._step = 0

  def _instruction_observation_space(self):
    return gym.spaces.Box(np.array([0]), np.array([0]), dtype=np.int)

  def _reward(self, instruction_state, action, original_reward):
    return original_reward, False

  def _generate_instructions(self, test=False):
    return np.array([0])

  def render(self):
    image = self.env.render()
    return image

class RoomRearrDreamWrapper(meta_exploration.MetaExplorationEnv):

  def __init__(self, env_id, wrapper, test=False):
    super().__init__(env_id, lambda state: state)
    self._env = room_rearr_mult_env.RoomRearrangementMultiWrapper(env_id)
    self.min_env_id = min(room_rearr_mult_env.RoomRearrangementMultiWrapper.TRAIN_SPEC_IDS)
    self.max_env_id = max(room_rearr_mult_env.RoomRearrangementMultiWrapper.TEST_SPEC_IDS)
    
  @classmethod
  def instruction_wrapper(cls):
    return InstructionWrapper

  @classmethod
  def create_env(cls, seed, test=False, wrapper=None):
    random = np.random.RandomState(seed)
    train_ids, test_ids = cls.env_ids()
    to_sample = test_ids if test else train_ids
    env_id = to_sample[random.randint(len(to_sample))]
    return cls(env_id, wrapper, test)

  def _step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = obs["rgb"]
    return obs, reward, done, info

  def _reset(self):
    # Don't set the seed, otherwise can cheat from initial camera angle position!
    obs = self._env.reset()
    obs = obs["rgb"]
    return obs

  @classmethod
  def env_ids(cls):
    train_ids, val_ids, test_ids = room_rearr_mult_env.RoomRearrangementMultiWrapper.get_task_id_space()
    return list(train_ids), list(test_ids)

  def _observation_space(self):
    obs_space = self._env.observation_space["rgb"]
    return obs_space.low, obs_space.high, obs_space.dtype

  def _env_id_space(self):
    low = np.array([0])
    high = np.array([self.max_env_id - self.min_env_id])
    dtype = np.int
    return low, high, dtype

  @property
  def observation_space(self):
    observation_low, observation_high, dtype = self._observation_space()
    env_id_low, env_id_high, dtype = self._env_id_space()
    return gym.spaces.Dict({
        "observation": gym.spaces.Box(
            observation_low, observation_high, dtype=dtype),
        "env_id": gym.spaces.Box(env_id_low, env_id_high, dtype=dtype)
    })

  @property
  def action_space(self):
    return self._env.action_space

  def render(self):
    return self._env.render("rgb")