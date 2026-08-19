# NetPulse — Passive Network Congestion Prediction from Packet Timing

NetPulse is a machine-learning-based network congestion prediction system that forecasts network congestion 2–5 seconds ahead using only inter-packet arrival times (IATs).

The system does not inspect packet payloads or application-level content. Instead, it analyzes temporal patterns in packet arrival timing to predict the upcoming congestion state.

The project combines a Temporal Convolutional Network (TCN), classical machine-learning baselines, synthetic network trace generation, model evaluation, and an interactive web dashboard.

---

# Overview

Network congestion can cause increased latency, packet delay, jitter, buffering, and degraded application performance.

Most traditional congestion monitoring approaches rely on packet-level information, application-level information, active probing, or network infrastructure metrics.

NetPulse investigates an alternative approach:

> Can temporal patterns in packet arrival times be used to predict upcoming network congestion without inspecting packet payloads?

The system receives inter-packet arrival times, divides them into temporal windows, and predicts one of three congestion states:

```text
Green
Yellow
Red
```

The primary model is a Temporal Convolutional Network that learns temporal patterns directly from IAT sequences.

Two baselines are provided for comparison:

1. Random Forest using eight statistical features.
2. A rule-based threshold model using IAT standard deviation.

---

# System Architecture

The overall NetPulse pipeline is:

```text
Packet Arrival Times
        |
        v
Inter-Packet Arrival Time (IAT)
        |
        v
Sliding Window
        |
        +-----------------------------+
        |                             |
        v                             v
    TCN Model                 Statistical Features
        |                             |
        |                    +--------+--------+
        |                    |                 |
        |                    v                 v
        |              Random Forest    Rule-Based Model
        |                    |                 |
        +--------------------+-----------------+
                             |
                             v
                    Congestion Prediction
                             |
                +------------+------------+
                |            |            |
                v            v            v
             Green        Yellow         Red
```

The TCN receives the raw IAT sequence.

The Random Forest receives eight statistical features extracted from the same IAT window.

The rule-based model uses the standard deviation of the IAT window and learned thresholds to determine the congestion state.

---

# Congestion Classes

NetPulse uses three congestion classes.

| Class | State | Description |
|---|---|---|
| 0 | Green | Low congestion and relatively stable packet timing |
| 1 | Yellow | Moderate congestion and increased timing variability |
| 2 | Red | Heavy congestion with significant timing variation and bursts |

---

# Synthetic Data Generation

NetPulse includes a synthetic network trace generator for creating controlled IAT sequences with congestion labels.

The generator creates different traffic conditions corresponding to the three congestion classes.

The generated traces can include characteristics such as:

- IAT mean
- IAT standard deviation
- Jitter
- Burst probability
- Burst length
- Retransmission-like long gaps
- Congestion state transitions

The congestion states evolve using a state-transition process rather than being completely independent observations.

This provides a controlled environment for training and evaluating the prediction models.

## Dataset

The current generated dataset contains:

```text
Total samples : 22,440
Training      : 17,952
Testing       : 4,488
```

Class distribution:

```text
Green  : 8,083
Yellow : 7,752
Red    : 6,605
```

The dataset is stored using NumPy arrays.

```text
data/
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
└── metadata.json
```

---

# Machine Learning Models

NetPulse evaluates three prediction approaches.

---

## 1. Temporal Convolutional Network

The Temporal Convolutional Network is the primary prediction model.

The TCN is designed to learn temporal relationships in the IAT sequence using causal and dilated convolutions.

The final temporal representation is passed to a linear classifier that produces three output classes.

---

## 2. Random Forest Baseline

The Random Forest provides a classical machine-learning baseline.

Instead of processing the complete IAT sequence directly, each window is converted into eight statistical features.

The extracted features are:

| Feature | Description |
|---|---|
| Mean IAT | Average inter-packet arrival time |
| Standard deviation | Overall variation in IAT |
| Jitter | Mean absolute change between consecutive IAT values |
| Minimum IAT | Minimum observed IAT |
| Maximum IAT | Maximum observed IAT |
| Skewness | Asymmetry of the IAT distribution |
| Kurtosis | Distribution tail characteristic |
| Coefficient of variation | Standard deviation normalized by the mean |

The resulting feature vector has the form:

```text
[
    mean_iat,
    std_iat,
    jitter,
    min_iat,
    max_iat,
    skewness,
    kurtosis,
    cov
]
```

The Random Forest is trained using these eight features.

Hyperparameter search is performed over parameters such as:

- Number of estimators
- Maximum tree depth
- Minimum samples required for splitting

Model selection uses cross-validation and macro-F1.

---

## 3. Rule-Based Baseline

The rule-based baseline provides a simple statistical comparison.

It uses the standard deviation of the IAT window as an indicator of congestion.

The logic is:

```text
Low IAT standard deviation
        |
        v
      Green

Medium IAT standard deviation
        |
        v
      Yellow

High IAT standard deviation
        |
        v
       Red
```

Two thresholds are learned from the training data.

The threshold search attempts to maximize macro-F1 on the training labels.

The learned values are stored in:

```text
saved_models/rule_info.json
```

---

# Feature Engineering

The Random Forest baseline uses eight statistical features extracted from each IAT window.

For every window, the feature extraction process calculates:

### 1. Mean IAT

```text
mean(w)
```

Represents the average packet arrival interval.

### 2. Standard Deviation

```text
std(w)
```

Represents overall variability in packet timing.

### 3. Jitter

```text
mean(abs(diff(w)))
```

Measures the average change between consecutive packet arrival intervals.

### 4. Minimum IAT

```text
min(w)
```

Represents the smallest observed packet interval.

### 5. Maximum IAT

```text
max(w)
```

Represents the largest observed packet interval.

### 6. Skewness

Skewness is calculated from the standardized third moment.

### 7. Kurtosis

Kurtosis is calculated from the standardized fourth moment.

### 8. Coefficient of Variation

```text
std(w) / mean(w)
```

This represents normalized variability in packet timing.

The resulting feature matrix has shape:

```text
(N, 8)
```

where `N` is the number of IAT windows.

---

# Training Pipeline

The training pipeline consists of the following stages:

```text
Dataset
   |
   v
Load or Generate Data
   |
   +-------------------------+
   |                         |
   v                         v
Feature Extraction       Raw IAT Windows
   |                         |
   v                         v
Random Forest             TCN
   |
   v
Rule Baseline
   |
   +-----------+-------------+
               |
               v
        Save Trained Models
               |
               v
       Save Training Results
```

The training orchestrator:

1. Loads the existing dataset if available.
2. Generates a dataset if one is not available.
3. Trains the rule-based baseline.
4. Extracts statistical features.
5. Trains the Random Forest.
6. Trains the TCN.
7. Saves model checkpoints and training results.

---

# Saved Models

The trained models are stored in:

```text
saved_models/
```



## TCN

```text
tcn.pt
```

Contains the trained PyTorch TCN weights.

## Random Forest

```text
rf.pkl
```

Contains the trained Random Forest model.

## Rule Model

```text
rule_info.json
```

Stores the learned rule-based thresholds and associated training information.

---

# Evaluation

The evaluation pipeline compares the three approaches on the held-out test set.

The models evaluated are:

```text
1. Rule Baseline
2. Random Forest
3. TCN
```

## Evaluation Metrics

The system reports:

- Macro-F1
- Macro precision
- Macro recall
- Confusion matrix
- Inference latency

Macro-F1 is particularly useful for comparing performance across the three congestion classes.

---

# Dashboard

NetPulse includes an interactive web dashboard for demonstrating the congestion prediction system.

The dashboard communicates with the Flask backend through API endpoints.

The main dashboard areas include:

- Data Lab
- Training Arena
- Evaluation
- Live Demo
- Scenario simulation
- Packet timing visualization
- Congestion prediction
- Early-warning metrics



## Evaluation

The Evaluation section compares the trained models.

The comparison can include:

- Model performance
- Macro-F1
- Precision
- Recall
- Confusion matrices
- Per-class performance
- Inference latency

The objective is to determine whether the temporal deep-learning model provides an advantage over simpler baselines.

---

The backend is implemented using Flask.

The API provides functionality for:

- Generating data
- Loading models
- Training models
- Evaluating models
- Predicting congestion
- Running live simulation
- Providing dashboard data

The backend runs on:

```text
http://localhost:5000
```

The frontend communicates with the backend through HTTP API requests.

---


# Running with Docker

## Prerequisites

Install:

- Docker Desktop
- Git
- A modern web browser

Make sure Docker Desktop is running before executing the commands below.

## Build the Image

Open a terminal in the project root:

```bash
docker build -t netpulse .
```

## Run the Container

```bash
docker run --rm -p 5000:5000 netpulse
```

The application will start on:

```text
http://localhost:5000
```

Open the URL in a browser.

## Stop the Application

Press:

```text
Ctrl + C
```

The container uses the `--rm` option, so the stopped container is automatically removed.

---
# Using the Dashboard

The recommended workflow is:

```text
1. Start the application
        |
        v
2. Open the dashboard
        |
        v
3. Generate or inspect data
        |
        v
4. Train models if required
        |
        v
5. Run evaluation
        |
        v
6. Start the live demonstration
```

The trained models are saved in:

```text
saved_models/
```

---



# Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Classical Machine Learning | scikit-learn |
| Numerical Computing | NumPy |
| Scientific Computing | SciPy |
| Backend | Flask |
| Frontend | HTML |
| Styling | CSS |
| Client Logic | JavaScript |
| Data Storage | NumPy `.npy` and JSON |
| Model Storage | PyTorch and pickle |
| Deployment | Docker |

---

# Limitations

The current implementation has several limitations.

### Synthetic Dataset

The primary experiments use synthetically generated network traces. Performance on synthetic data does not automatically imply equivalent performance on real network traffic.

### Limited Input Information

The system relies only on packet timing information and therefore does not use other potentially useful network features such as:

- Packet size
- Flow identifiers
- Transport-level information
- Queue occupancy
- Link utilization
- Explicit congestion signals

### Generalization

The ability to generalize across networks, applications, protocols, and traffic patterns requires further evaluation.

### Congestion Labels

Synthetic labels are generated from the simulated congestion conditions. Real-world evaluation requires reliable ground-truth congestion labels.

ssive packet-timing information can provide useful early congestion predictions without requiring access to packet payloads.
