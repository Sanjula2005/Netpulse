"""
NetPulse — Random Forest Baseline
===================================
Uses 8 hand-crafted statistical features per IAT window.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features import extract_features


class RFBaseline:
    def __init__(self):
        self.model = None

    def train(self, X_windows, y_labels):
        """Train with grid-search over key hyperparameters."""
        X_feat = extract_features(X_windows)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5],
        }
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid, cv=3, scoring="f1_macro", n_jobs=-1,
        )
        gs.fit(X_feat, y_labels)
        self.model = gs.best_estimator_
        return {"best_params": gs.best_params_,
                "best_cv_f1": round(float(gs.best_score_), 4)}

    def predict(self, X_windows):
        return self.model.predict(extract_features(X_windows))

    def predict_proba(self, X_windows):
        return self.model.predict_proba(extract_features(X_windows))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
