"""
NetPulse — Training Orchestrator
==================================
Trains TCN, Random Forest, and Rule-based models. Saves checkpoints + metrics.
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import CongestionDataGenerator
from models.tcn import build_tcn
from models.baseline_rf import RFBaseline
from models.baseline_rule import RuleBaseline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def ensure_data(num_traces=40, window_size=100, stride=10, seed=42):
    """Generate dataset if not present."""
    if os.path.exists(os.path.join(DATA_DIR, "X_train.npy")):
        return json.load(open(os.path.join(DATA_DIR, "metadata.json")))
    gen = CongestionDataGenerator(seed=seed)
    return gen.save_dataset(DATA_DIR, num_traces=num_traces,
                            window_size=window_size, stride=stride)


def load_data():
    """Load train/test splits from disk."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_train, y_train, X_test, y_test


# ── TCN Training ──────────────────────────────────────────────────────

def train_tcn(X_train, y_train, X_test, y_test,
              epochs=30, batch_size=64, lr=1e-3, config=None,
              progress_callback=None):
    """Train the TCN model. Returns model + training history."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalise IATs (z-score per-window)
    mean = X_train.mean(axis=1, keepdims=True)
    std = X_train.std(axis=1, keepdims=True) + 1e-8
    X_tr_norm = (X_train - mean) / std

    mean_te = X_test.mean(axis=1, keepdims=True)
    std_te = X_test.std(axis=1, keepdims=True) + 1e-8
    X_te_norm = (X_test - mean_te) / std_te

    # Reshape: (N, W) → (N, 1, W)
    Xt = torch.tensor(X_tr_norm).unsqueeze(1).to(device)
    yt = torch.tensor(y_train).to(device)
    Xv = torch.tensor(X_te_norm).unsqueeze(1).to(device)
    yv = torch.tensor(y_test).to(device)

    train_ds = TensorDataset(Xt, yt)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = build_tcn(config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = loss_fn(val_logits, yv).item()
            val_acc = (val_logits.argmax(1) == yv).float().mean().item()

        tl = round(epoch_loss / len(Xt), 4)
        history["train_loss"].append(tl)
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"  Epoch {epoch+1}/{epochs}  loss={tl}  val_acc={round(val_acc,4)}")

        if progress_callback:
            progress_callback(epoch + 1, epochs, history)

    # Save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "tcn.pt"))
    with open(os.path.join(MODEL_DIR, "tcn_history.json"), "w") as f:
        json.dump(history, f)

    return model, history


# ── RF Training ───────────────────────────────────────────────────────

def train_rf(X_train, y_train):
    """Train Random Forest baseline."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    rf = RFBaseline()
    info = rf.train(X_train, y_train)
    rf.save(os.path.join(MODEL_DIR, "rf.pkl"))
    with open(os.path.join(MODEL_DIR, "rf_info.json"), "w") as f:
        json.dump(info, f)
    return rf, info


# ── Rule Training ─────────────────────────────────────────────────────

def train_rule(X_train, y_train):
    """Train rule-based baseline (threshold search)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    rule = RuleBaseline()
    info = rule.train(X_train, y_train)
    with open(os.path.join(MODEL_DIR, "rule_info.json"), "w") as f:
        json.dump(info, f)
    return rule, info


# ── Full Pipeline ─────────────────────────────────────────────────────

def train_all(epochs=15, batch_size=64, lr=1e-3, num_traces=40,
              window_size=100, progress_callback=None):
    """Run the complete training pipeline for all three models."""
    print("=" * 60)
    print("  NetPulse - Training Pipeline")
    print("=" * 60)

    # 1. Data
    print("\n[1/4] Generating data...")
    ensure_data(num_traces=num_traces, window_size=window_size)
    X_train, y_train, X_test, y_test = load_data()
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    results = {}

    # 2. Rule baseline
    print("\n[2/4] Training Rule-based baseline...")
    t0 = time.time()
    rule, rule_info = train_rule(X_train, y_train)
    results["rule"] = {**rule_info, "train_time": round(time.time() - t0, 2)}
    print(f"  Thresholds: mid={rule_info['thresh_mid']}, high={rule_info['thresh_high']}")

    # 3. Random Forest
    print("\n[3/4] Training Random Forest...")
    t0 = time.time()
    rf, rf_info = train_rf(X_train, y_train)
    results["rf"] = {**rf_info, "train_time": round(time.time() - t0, 2)}
    print(f"  Best CV F1: {rf_info['best_cv_f1']}")

    # 4. TCN
    print(f"\n[4/4] Training TCN ({epochs} epochs)...")
    t0 = time.time()
    tcn, tcn_hist = train_tcn(X_train, y_train, X_test, y_test,
                               epochs=epochs, batch_size=batch_size, lr=lr,
                               progress_callback=progress_callback)
    results["tcn"] = {"history": tcn_hist,
                      "final_val_acc": tcn_hist["val_acc"][-1],
                      "train_time": round(time.time() - t0, 2)}
    print(f"  Final val acc: {tcn_hist['val_acc'][-1]}")

    with open(os.path.join(MODEL_DIR, "train_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n[OK] All models trained and saved to saved_models/")
    return results


if __name__ == "__main__":
    train_all(epochs=15)
