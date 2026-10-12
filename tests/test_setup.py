"""Smoke tests confirming the development environment is set up correctly."""

import sys

import gymnasium as gym
import numpy as np


def test_python_version():
    assert sys.version_info[:2] == (3, 12)


def test_gymnasium_spaces_are_seedable():
    space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
    space.seed(0)
    first = space.sample()
    space.seed(0)
    assert np.array_equal(first, space.sample())
