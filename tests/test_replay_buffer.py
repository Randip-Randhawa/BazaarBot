import numpy as np

from bazaar.env import OBS_SIZE
from bazaar.replay_buffer import ReplayBuffer


def add_transitions(buffer, count):
    """Add `count` transitions where every field is derived from the index i,
    so a sampled row can be traced back to the transition it came from."""
    for i in range(count):
        state = np.full(OBS_SIZE, i, dtype=np.float32)
        next_state = state + 0.5
        buffer.add(state, i, float(i), next_state, i % 2 == 0)


def test_add_and_sample():
    buffer = ReplayBuffer(capacity=10, seed=0)
    add_transitions(buffer, 5)
    assert len(buffer) == 5

    states, actions, rewards, next_states, terminated = buffer.sample(5)

    # Every stored transition comes back, with its fields still lined up.
    assert sorted(rewards) == [0, 1, 2, 3, 4]
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, terminated):
        assert np.all(state == reward)
        assert np.all(next_state == reward + 0.5)
        assert action == reward
        assert done == (action % 2 == 0)


def test_batch_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=10, seed=0)
    add_transitions(buffer, 10)

    states, actions, rewards, next_states, terminated = buffer.sample(4)

    assert states.shape == next_states.shape == (4, OBS_SIZE)
    assert actions.shape == rewards.shape == terminated.shape == (4,)
    assert states.dtype == next_states.dtype == np.float32
    assert actions.dtype == np.int64
    assert rewards.dtype == terminated.dtype == np.float32


def test_same_seed_gives_same_batches():
    first = ReplayBuffer(capacity=10, seed=7)
    second = ReplayBuffer(capacity=10, seed=7)
    add_transitions(first, 10)
    add_transitions(second, 10)

    for _ in range(3):
        for a, b in zip(first.sample(4), second.sample(4)):
            assert np.array_equal(a, b)


def test_oldest_transitions_are_overwritten_when_full():
    buffer = ReplayBuffer(capacity=3, seed=0)
    add_transitions(buffer, 5)
    assert len(buffer) == 3

    _, _, rewards, _, _ = buffer.sample(3)
    assert sorted(rewards) == [2, 3, 4]


def test_stored_observation_is_a_copy():
    buffer = ReplayBuffer(capacity=1, seed=0)
    state = np.zeros(OBS_SIZE, dtype=np.float32)
    next_state = np.ones(OBS_SIZE, dtype=np.float32)
    buffer.add(state, 0, 0.0, next_state, False)

    state[:] = 9.0
    next_state[:] = 9.0

    states, _, _, next_states, _ = buffer.sample(1)
    assert np.all(states[0] == 0.0)
    assert np.all(next_states[0] == 1.0)


def test_sampling_is_without_replacement():
    buffer = ReplayBuffer(capacity=20, seed=0)
    add_transitions(buffer, 20)

    for _ in range(10):
        _, _, rewards, _, _ = buffer.sample(20)
        assert len(set(rewards)) == 20
