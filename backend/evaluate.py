"""
NetPulse — Model Evaluation
=============================
Compares all three models on the test set with standard metrics.
"""

import os, sys, json, time
import numpy as np
import torch
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(__file__))
from models.tcn import build_tcn
from models.baseline_rf import RFBaseline
from models.baseline_rule import RuleBaseline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
LABELS = ["green", "yellow", "red"]


def load_test_data():
    X = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X, y


def _measure_latency(predict_fn, X_sample, n=100):
    """Measure average inference latency in ms."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        predict_fn(X_sample)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 3)


def _metrics(y_true, y_pred):
    return {
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class": json.loads(
            classification_report(y_true, y_pred, target_names=LABELS,
                                  output_dict=True, zero_division=0).__repr__()
        ) if False else _per_class(y_true, y_pred),
    }


def _per_class(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=LABELS,
                                   output_dict=True, zero_division=0)
    return {name: {k: round(v, 4) for k, v in vals.items()}
            for name, vals in report.items() if name in LABELS}


def evaluate_all():
    """Evaluate all three models and return comparison dict."""
    X_test, y_test = load_test_data()
    sample = X_test[:10]  # for latency measurement
    results = {}

    # ── Rule baseline ─────────────────────────────────────────────
    rule_info = json.load(open(os.path.join(MODEL_DIR, "rule_info.json")))
    rule = RuleBaseline()
    rule.thresh_mid = rule_info["thresh_mid"]
    rule.thresh_high = rule_info["thresh_high"]
    y_rule = rule.predict(X_test)
    results["rule"] = {
        **_metrics(y_test, y_rule),
        "latency_ms": _measure_latency(lambda x: rule.predict(x), sample),
    }

    # ── Random Forest ─────────────────────────────────────────────
    rf = RFBaseline()
    rf.load(os.path.join(MODEL_DIR, "rf.pkl"))
    y_rf = rf.predict(X_test)
    results["rf"] = {
        **_metrics(y_test, y_rf),
        "latency_ms": _measure_latency(lambda x: rf.predict(x), sample),
    }

    # ── TCN ───────────────────────────────────────────────────────
    device = torch.device("cpu")
    model = build_tcn().to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "tcn.pt"),
                                     map_location=device, weights_only=True))
    model.eval()

    def tcn_predict(windows):
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True) + 1e-8
        normed = (windows - mean) / std
        t = torch.tensor(normed).unsqueeze(1).float()
        with torch.no_grad():
            return model(t).argmax(1).numpy()

    y_tcn = tcn_predict(X_test)
    results["tcn"] = {
        **_metrics(y_test, y_tcn),
        "latency_ms": _measure_latency(lambda x: tcn_predict(x), sample),
    }

    # ── Summary ───────────────────────────────────────────────────
    results["summary"] = {
        "tcn_beats_rf_by": round(results["tcn"]["f1_macro"] -
                                  results["rf"]["f1_macro"], 4),
        "tcn_beats_rule_by": round(results["tcn"]["f1_macro"] -
                                    results["rule"]["f1_macro"], 4),
        "best_model": max(results, key=lambda k: results[k].get("f1_macro", 0)
                          if k != "summary" else 0),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = evaluate_all()
    print(json.dumps(res, indent=2))
