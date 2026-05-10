"""
NeuraX - Synthetic EEG Signal Generator
Simulates realistic EEG signals for 4 mental commands:
  0 = idle, 1 = cursor_left, 2 = cursor_right, 3 = select
"""

import numpy as np


# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 50),
}

COMMANDS = {0: "idle", 1: "cursor_left", 2: "cursor_right", 3: "select"}

SAMPLE_RATE = 256  # Hz
WINDOW_DURATION = 1.0  # seconds
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)


def _sine(freq, t, amplitude=1.0, phase=0.0):
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def generate_eeg_window(command: int, noise_level: float = 0.3) -> np.ndarray:
    """
    Generate a single 1-second EEG window for a given command.
    Returns array of shape (WINDOW_SAMPLES,)
    """
    t = np.linspace(0, WINDOW_DURATION, WINDOW_SAMPLES, endpoint=False)
    signal = np.zeros(WINDOW_SAMPLES)

    rng = np.random.default_rng()

    if command == 0:  # idle: dominant alpha
        signal += _sine(10, t, amplitude=2.0, phase=rng.uniform(0, np.pi))
        signal += _sine(11, t, amplitude=1.5, phase=rng.uniform(0, np.pi))
        signal += _sine(6,  t, amplitude=0.5, phase=rng.uniform(0, np.pi))

    elif command == 1:  # cursor_left: left motor imagery - beta suppression, mu rhythm
        signal += _sine(10, t, amplitude=0.8, phase=rng.uniform(0, np.pi))  # suppressed alpha
        signal += _sine(20, t, amplitude=1.8, phase=rng.uniform(0, np.pi))  # beta burst
        signal += _sine(12, t, amplitude=1.2, phase=rng.uniform(0, np.pi))

    elif command == 2:  # cursor_right: right motor imagery
        signal += _sine(10, t, amplitude=0.7, phase=rng.uniform(0, np.pi))
        signal += _sine(22, t, amplitude=2.0, phase=rng.uniform(0, np.pi))
        signal += _sine(15, t, amplitude=1.0, phase=rng.uniform(0, np.pi))

    elif command == 3:  # select: mental click - P300-like response + gamma
        signal += _sine(5,  t, amplitude=1.5, phase=rng.uniform(0, np.pi))  # theta
        signal += _sine(40, t, amplitude=1.2, phase=rng.uniform(0, np.pi))  # gamma
        signal += _sine(8,  t, amplitude=1.0, phase=rng.uniform(0, np.pi))

    # Add pink noise (realistic EEG noise)
    white_noise = rng.normal(0, noise_level, WINDOW_SAMPLES)
    pink_noise = np.cumsum(white_noise) * 0.01
    signal += pink_noise

    return signal.astype(np.float32)


def generate_dataset(samples_per_class: int = 300, noise_level: float = 0.3):
    """
    Generate a balanced EEG dataset.
    Returns X of shape (N, WINDOW_SAMPLES), y of shape (N,)
    """
    X, y = [], []
    for cmd in COMMANDS:
        for _ in range(samples_per_class):
            X.append(generate_eeg_window(cmd, noise_level))
            y.append(cmd)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


if __name__ == "__main__":
    X, y = generate_dataset(samples_per_class=10)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y, return_counts=True)}")
