import numpy as np

from OffDRL.buffer import ReplayBuffer


def test_replay_buffer_load_and_sample():
    buffer = ReplayBuffer(
        buffer_size=32,
        obs_shape=(4,),
        obs_dtype=np.float32,
        action_shape=(2,),
        action_dtype=np.float32,
        device="cpu",
    )

    dataset = {
        "observations": np.random.randn(16, 4).astype(np.float32),
        "next_observations": np.random.randn(16, 4).astype(np.float32),
        "actions": np.random.randn(16, 2).astype(np.float32),
        "rewards": np.random.randn(16).astype(np.float32),
        "terminateds": np.random.randint(0, 2, size=(16,)).astype(np.float32),
        "truncateds": np.random.randint(0, 2, size=(16,)).astype(np.float32),
    }

    buffer.load_dataset(dataset)
    batch = buffer.sample(batch_size=8)

    assert buffer._size == 16
    assert batch["observations"].shape == (8, 4)
    assert batch["actions"].shape == (8, 2)
