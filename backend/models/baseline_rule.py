"""
NetPulse — Rolling Mean Threshold Baseline
============================================
Industry-standard rule-based congestion detector.
Uses IAT standard deviation thresholds (higher variance = more congestion).
"""

import numpy as np
from sklearn.metrics import f1_score


class RuleBaseline:
    def __init__(self):
        self.thresh_mid = None
        self.thresh_high = None

    def train(self, X_windows, y_labels):
        """Brute-force search for optimal std-deviation thresholds."""
        stds = np.array([np.std(w) for w in X_windows])
        best_f1, best = 0, (100, 250)

        for mid in np.arange(50, 350, 25):
            for high in np.arange(mid + 50, 600, 25):
                preds = np.where(stds > high, 2, np.where(stds > mid, 1, 0))
                f1 = f1_score(y_labels, preds, average="macro")
                if f1 > best_f1:
                    best_f1, best = f1, (mid, high)

        self.thresh_mid, self.thresh_high = best
        return {"thresh_mid": float(self.thresh_mid),
                "thresh_high": float(self.thresh_high),
                "best_f1": round(float(best_f1), 4)}

    def predict(self, X_windows):
        stds = np.array([np.std(w) for w in X_windows])
        return np.where(stds > self.thresh_high, 2,
                        np.where(stds > self.thresh_mid, 1, 0))

    def predict_proba(self, X_windows):
        """Fake probabilities based on distance from thresholds."""
        stds = np.array([np.std(w) for w in X_windows])
        probs = np.zeros((len(stds), 3))
        for i, s in enumerate(stds):
            if s > self.thresh_high:
                probs[i] = [0.05, 0.15, 0.80]
            elif s > self.thresh_mid:
                probs[i] = [0.15, 0.70, 0.15]
            else:
                probs[i] = [0.80, 0.15, 0.05]
        return probs
