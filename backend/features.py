import numpy as np

FEATURE_NAMES = [
    "mean_iat", "std_iat", "jitter", "min_iat",
    "max_iat", "skewness", "kurtosis", "cov",
]


def extract_features(windows: np.ndarray) -> np.ndarray:
    """
    Extract 8 statistical features from each IAT window.

    Args:
        windows: array of shape (N, W) — N windows of W IAT values

    Returns:
        features: array of shape (N, 8)
    """

    feats = []

    for w in windows:
        diffs = np.diff(w)

        mean = np.mean(w)
        std = np.std(w) + 1e-8

        skewness = np.mean(((w - mean) / std) ** 3)
        kurtosis = np.mean(((w - mean) / std) ** 4) - 3

        feats.append([
            mean,
            np.std(w),
            np.mean(np.abs(diffs)),
            np.min(w),
            np.max(w),
            float(skewness),
            float(kurtosis),
            np.std(w) / (mean + 1e-8),
        ])

    return np.array(feats, dtype=np.float32)