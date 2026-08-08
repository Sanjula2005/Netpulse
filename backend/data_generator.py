"""
NetPulse — Synthetic Network Trace Generator
=============================================
Generates realistic IAT (Inter-Arrival Time) sequences with congestion labels.
No NS-3 or Mininet required — pure Python statistical simulation.

Congestion states:
  0 (GREEN)  — Low load, steady IATs
  1 (YELLOW) — Moderate congestion, increased jitter
  2 (RED)    — Heavy congestion, bursty + retransmission patterns
"""

import numpy as np
import json
import os


class CongestionDataGenerator:
    STATE_NAMES = {0: "green", 1: "yellow", 2: "red"}

    # Markov transition matrix — controls how congestion evolves
    TRANSITION_PROBS = {
        0: [0.70, 0.25, 0.05],
        1: [0.20, 0.50, 0.30],
        2: [0.05, 0.35, 0.60],
    }

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.state_params = {
            0: {"iat_mean": 1000, "iat_std": 150, "jitter": 0.05,
                "burst_prob": 0.01, "burst_len": (2, 5)},
            1: {"iat_mean": 400, "iat_std": 200, "jitter": 0.15,
                "burst_prob": 0.05, "burst_len": (3, 10)},
            2: {"iat_mean": 150, "iat_std": 300, "jitter": 0.35,
                "burst_prob": 0.15, "burst_len": (5, 20)},
        }

    def _state_sequence(self, length, seg_mean=200):
        """Generate Markov-chain congestion state sequence."""
        states = []
        cur = self.rng.choice(3, p=[0.5, 0.3, 0.2])
        while len(states) < length:
            seg = max(50, int(self.rng.exponential(seg_mean)))
            seg = min(seg, length - len(states))
            states.extend([cur] * seg)
            cur = self.rng.choice(3, p=self.TRANSITION_PROBS[cur])
        return np.array(states[:length])

    def _iats_for_state(self, state, n):
        """Sample IATs from the distribution of a given congestion state."""
        p = self.state_params[state]
        iats = self.rng.exponential(p["iat_mean"], size=n)
        iats += self.rng.normal(0, p["iat_mean"] * p["jitter"], size=n)

        # Inject micro-bursts (clusters of tiny IATs)
        for i in np.where(self.rng.random(n) < p["burst_prob"])[0]:
            end = min(i + self.rng.integers(*p["burst_len"]), n)
            iats[i:end] = self.rng.exponential(20, size=end - i)

        # RED: add retransmission-like long gaps
        if state == 2:
            retx = self.rng.random(n) < 0.03
            iats[retx] = self.rng.exponential(5000, size=retx.sum())

        return np.maximum(iats, 1.0)

    def generate_trace(self, length=2000, seg_mean=200):
        """Generate one trace → (iats, states) arrays."""
        states = self._state_sequence(length, seg_mean)
        iats = np.zeros(length)
        for s in range(3):
            mask = states == s
            if mask.any():
                iats[mask] = self._iats_for_state(s, mask.sum())
        return iats.astype(np.float32), states.astype(np.int64)

    def generate_dataset(self, num_traces=100, trace_len=2000,
                         window_size=100, stride=10):
        """Create windowed dataset for ML. Label = dominant state 2-5s ahead."""
        X, y = [], []
        for _ in range(num_traces):
            iats, states = self.generate_trace(trace_len)
            for start in range(0, len(iats) - window_size - 30, stride):
                window = iats[start:start + window_size]
                future = states[start + window_size:start + window_size + 30]
                label = int(np.bincount(future).argmax())
                X.append(window)
                y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def save_dataset(self, output_dir, num_traces=100, trace_len=2000,
                     window_size=100, stride=10, test_split=0.2):
        """Generate, split, and save train/test datasets as .npy files."""
        os.makedirs(output_dir, exist_ok=True)
        X, y = self.generate_dataset(num_traces, trace_len, window_size, stride)

        idx = self.rng.permutation(len(X))
        X, y = X[idx], y[idx]
        split = int(len(X) * (1 - test_split))

        np.save(os.path.join(output_dir, "X_train.npy"), X[:split])
        np.save(os.path.join(output_dir, "y_train.npy"), y[:split])
        np.save(os.path.join(output_dir, "X_test.npy"), X[split:])
        np.save(os.path.join(output_dir, "y_test.npy"), y[split:])

        meta = {
            "num_traces": num_traces, "trace_len": trace_len,
            "window_size": window_size, "stride": stride,
            "total_samples": len(X), "train": split, "test": len(X) - split,
            "class_dist": {
                "green": int((y == 0).sum()),
                "yellow": int((y == 1).sum()),
                "red": int((y == 2).sum()),
            },
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta


if __name__ == "__main__":
    gen = CongestionDataGenerator(seed=42)
    meta = gen.save_dataset("../data", num_traces=120, window_size=100)
    print("Dataset generated:", json.dumps(meta, indent=2))
