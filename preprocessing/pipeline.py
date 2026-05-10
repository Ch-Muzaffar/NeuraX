"""
NeuraX - EEG Preprocessing Pipeline
Handles: bandpass filtering, notch filtering, normalization,
         feature extraction (FFT + statistical features)
"""

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, sosfiltfilt
from scipy.fft import rfft, rfftfreq

SAMPLE_RATE = 256  # Hz
NYQUIST = SAMPLE_RATE / 2


# ─── Filters ────────────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, low: float = 0.5, high: float = 50.0) -> np.ndarray:
    """
    Apply a 4th-order Butterworth bandpass filter.
    Keeps only the frequency range relevant for EEG (0.5 - 50 Hz).
    """
    sos = butter(4, [low / NYQUIST, high / NYQUIST], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def notch_filter(signal: np.ndarray, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """
    Remove power line noise (50 Hz in Pakistan/Europe, 60 Hz in USA).
    """
    b, a = iirnotch(freq / NYQUIST, quality)
    from scipy.signal import lfilter
    return lfilter(b, a, signal).astype(np.float32)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalization per window."""
    mean = signal.mean()
    std = signal.std()
    if std < 1e-8:
        return signal - mean
    return ((signal - mean) / std).astype(np.float32)


# ─── Feature Extraction ──────────────────────────────────────────────────────

def extract_band_powers(signal: np.ndarray) -> dict:
    """
    Compute average power in each EEG frequency band using FFT.
    Returns dict with band name -> power value.
    """
    n = len(signal)
    freqs = rfftfreq(n, d=1.0 / SAMPLE_RATE)
    fft_magnitude = np.abs(rfft(signal)) ** 2 / n

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 50),
    }

    powers = {}
    for band, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        powers[band] = float(fft_magnitude[mask].mean()) if mask.any() else 0.0

    return powers


def extract_statistical_features(signal: np.ndarray) -> dict:
    """
    Time-domain statistical features.
    """
    return {
        "mean":     float(np.mean(signal)),
        "std":      float(np.std(signal)),
        "var":      float(np.var(signal)),
        "skew":     float(_skew(signal)),
        "kurtosis": float(_kurtosis(signal)),
        "rms":      float(np.sqrt(np.mean(signal ** 2))),
        "peak":     float(np.max(np.abs(signal))),
    }


def _skew(x: np.ndarray) -> float:
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-8:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-8:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3)


def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Full feature vector for one EEG window.
    Combines band powers + statistical features.
    Returns 1D numpy array of length 12.
    """
    clean = bandpass_filter(signal)
    clean = notch_filter(clean)
    clean = normalize(clean)

    band_feats = extract_band_powers(clean)
    stat_feats = extract_statistical_features(clean)

    feature_vector = list(band_feats.values()) + list(stat_feats.values())
    return np.array(feature_vector, dtype=np.float32)


def preprocess_dataset(X: np.ndarray) -> np.ndarray:
    """
    Apply extract_features to every sample in the dataset.
    Input:  X of shape (N, window_samples)
    Output: X of shape (N, n_features)
    """
    return np.array([extract_features(row) for row in X], dtype=np.float32)


if __name__ == "__main__":
    # Quick test
    from data.signal_generator import generate_eeg_window
    raw = generate_eeg_window(command=1)
    features = extract_features(raw)
    print(f"Feature vector length: {len(features)}")
    print(f"Features: {features}")
