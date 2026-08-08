"""
NetPulse — Flask API Server
==============================
REST API for the congestion prediction dashboard.
Endpoints: /api/generate, /api/train, /api/evaluate, /api/predict, /api/simulate, /api/status
"""

import os, sys, json, time, threading
import numpy as np
import torch
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import CongestionDataGenerator
from models.tcn import build_tcn
from models.baseline_rf import RFBaseline
from models.baseline_rule import RuleBaseline
from features import extract_features

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")

# ── Global state ──────────────────────────────────────────────────────
training_status = {"running": False, "progress": 0, "total": 0,
                   "phase": "idle", "history": None, "results": None}
loaded_models = {"tcn": None, "rf": None, "rule": None}


def _load_models():
    """Load saved models into memory."""
    try:
        model = build_tcn()
        model.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "tcn.pt"),
            map_location="cpu", weights_only=True))
        model.eval()
        loaded_models["tcn"] = model
    except Exception:
        pass

    # try:
    #     rf = RFBaseline()
    #     rf.load(os.path.join(MODEL_DIR, "rf.pkl"))
    #     loaded_models["rf"] = rf
    # except Exception:
    #     pass
        # RF disabled temporarily - rf.pkl hangs during loading
    loaded_models["rf"] = None
    
    try:
        info = json.load(open(os.path.join(MODEL_DIR, "rule_info.json")))
        rule = RuleBaseline()
        rule.thresh_mid = info["thresh_mid"]
        rule.thresh_high = info["thresh_high"]
        loaded_models["rule"] = rule
    except Exception:
        pass


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


@app.route("/api/status")
def status():
    models_ready = {k: v is not None for k, v in loaded_models.items()}
    data_ready = os.path.exists(os.path.join(DATA_DIR, "X_train.npy"))
    meta = None
    if data_ready:
        try:
            meta = json.load(open(os.path.join(DATA_DIR, "metadata.json")))
        except Exception:
            pass
    eval_results = None
    if os.path.exists(os.path.join(MODEL_DIR, "eval_results.json")):
        try:
            eval_results = json.load(open(os.path.join(MODEL_DIR, "eval_results.json")))
        except Exception:
            pass
    return jsonify({
        "models_ready": models_ready,
        "data_ready": data_ready,
        "data_meta": meta,
        "training": training_status,
        "eval_results": eval_results,
    })


@app.route("/api/generate", methods=["POST"])
def generate_data():
    """Generate synthetic dataset with custom parameters."""
    params = request.get_json(silent=True) or {}
    num_traces = params.get("num_traces", 120)
    window_size = params.get("window_size", 100)
    stride = params.get("stride", 10)
    seed = params.get("seed", 42)

    gen = CongestionDataGenerator(seed=seed)
    meta = gen.save_dataset(DATA_DIR, num_traces=num_traces,
                            window_size=window_size, stride=stride)
    return jsonify({"status": "ok", "metadata": meta})


@app.route("/api/train", methods=["POST"])
def train_models():
    """Start training all models in background."""
    if training_status["running"]:
        return jsonify({"error": "Training already in progress"}), 409

    params = request.get_json(silent=True) or {}
    epochs = params.get("epochs", 30)
    batch_size = params.get("batch_size", 64)
    lr = params.get("lr", 1e-3)

    def _run():
        training_status.update(running=True, progress=0, total=epochs,
                               phase="preparing", history=None, results=None)
        try:
            from train import ensure_data, load_data, train_tcn, train_rf, train_rule

            training_status["phase"] = "generating_data"
            ensure_data()
            X_train, y_train, X_test, y_test = load_data()

            # Rule
            training_status["phase"] = "training_rule"
            rule, rule_info = train_rule(X_train, y_train)

            # RF
            training_status["phase"] = "training_rf"
            rf, rf_info = train_rf(X_train, y_train)

            # TCN
            training_status["phase"] = "training_tcn"

            def cb(ep, total, hist):
                training_status["progress"] = ep
                training_status["total"] = total
                training_status["history"] = hist

            tcn, tcn_hist = train_tcn(X_train, y_train, X_test, y_test,
                                       epochs=epochs, batch_size=batch_size,
                                       lr=lr, progress_callback=cb)

            # Evaluate
            training_status["phase"] = "evaluating"
            from evaluate import evaluate_all
            eval_res = evaluate_all()

            training_status["results"] = eval_res
            training_status["phase"] = "done"

            # Reload models
            _load_models()

        except Exception as e:
            training_status["phase"] = f"error: {str(e)}"
        finally:
            training_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/evaluate")
def get_evaluation():
    """Return cached evaluation results."""
    path = os.path.join(MODEL_DIR, "eval_results.json")
    if not os.path.exists(path):
        return jsonify({"error": "No evaluation results. Train models first."}), 404
    return jsonify(json.load(open(path)))


@app.route("/api/predict", methods=["POST"])
def predict():
    """Real-time prediction on a single IAT window."""
    data = request.get_json()
    window = np.array(data["window"], dtype=np.float32)
    results = {}

    # Normalise for TCN
    if loaded_models["tcn"]:
        mean, std = window.mean(), window.std() + 1e-8
        normed = (window - mean) / std
        t = torch.tensor(normed).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = torch.softmax(loaded_models["tcn"](t), dim=1)[0]
        results["tcn"] = {
            "prediction": int(probs.argmax()),
            "probabilities": probs.tolist(),
        }

    if loaded_models["rf"]:
        probs = loaded_models["rf"].predict_proba(window.reshape(1, -1))[0]
        results["rf"] = {
            "prediction": int(np.argmax(probs)),
            "probabilities": probs.tolist(),
        }

    if loaded_models["rule"]:
        pred = int(loaded_models["rule"].predict(window.reshape(1, -1))[0])
        probs = loaded_models["rule"].predict_proba(window.reshape(1, -1))[0]
        results["rule"] = {
            "prediction": pred,
            "probabilities": probs.tolist(),
        }

    return jsonify(results)


@app.route("/api/simulate")
def simulate():
    """SSE stream: live simulated traffic with real-time predictions."""
    def stream():
        gen = CongestionDataGenerator(seed=int(time.time()) % 10000)
        iats, states = gen.generate_trace(length=5000, seg_mean=150)
        window_size = 100
        idx = window_size

        while idx < len(iats):
            window = iats[idx - window_size:idx]
            true_state = int(states[min(idx + 15, len(states) - 1)])

            preds = {}
            if loaded_models["tcn"]:
                mean, std = window.mean(), window.std() + 1e-8
                normed = (window - mean) / std
                t = torch.tensor(normed).unsqueeze(0).unsqueeze(0).float()
                with torch.no_grad():
                    p = torch.softmax(loaded_models["tcn"](t), dim=1)[0].tolist()
                preds["tcn"] = {"pred": int(np.argmax(p)), "probs": p}

            if loaded_models["rf"]:
                p = loaded_models["rf"].predict_proba(window.reshape(1, -1))[0].tolist()
                preds["rf"] = {"pred": int(np.argmax(p)), "probs": p}

            if loaded_models["rule"]:
                pred = int(loaded_models["rule"].predict(window.reshape(1, -1))[0])
                p = loaded_models["rule"].predict_proba(window.reshape(1, -1))[0].tolist()
                preds["rule"] = {"pred": pred, "probs": p}

            payload = json.dumps({
                "idx": int(idx),
                "iat": float(iats[idx - 1]),
                "window_mean": float(window.mean()),
                "window_std": float(window.std()),
                "true_state": true_state,
                "predictions": preds,
            })
            yield f"data: {payload}\n\n"
            idx += 1
            time.sleep(0.05)  # ~20 updates/sec

        yield "data: {\"done\": true}\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/sample_trace")
def sample_trace():
    """Generate a single trace for visualization."""
    length = request.args.get("length", 500, type=int)
    seed = request.args.get("seed", 42, type=int)
    gen = CongestionDataGenerator(seed=seed)
    iats, states = gen.generate_trace(length=length)
    return jsonify({
        "iats": iats.tolist(),
        "states": states.tolist(),
    })


# ── Boot ──────────────────────────────────────────────────────────────

_load_models()

if __name__ == "__main__":
    print("\n  [*] NetPulse -- Network Congestion Predictor")
    print("  Dashboard: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
