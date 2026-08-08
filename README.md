# NetPulse — Passive Network Congestion Prediction from Packet Timing

NetPulse is a machine-learning-based network congestion prediction system that forecasts network congestion 2–5 seconds ahead using only inter-packet arrival times (IATs).

The system does not inspect packet payloads or application-level content. Instead, it analyzes temporal patterns in packet arrival timing to predict the upcoming congestion state.

The project combines a Temporal Convolutional Network (TCN), classical machine-learning baselines, synthetic network trace generation, model evaluation, and an interactive web dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Input and Output](#input-and-output)
- [Congestion Classes](#congestion-classes)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Machine Learning Models](#machine-learning-models)
  - [Temporal Convolutional Network](#1-temporal-convolutional-network)
  - [Random Forest Baseline](#2-random-forest-baseline)
  - [Rule-Based Baseline](#3-rule-based-baseline)
- [Feature Engineering](#feature-engineering)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running with Docker](#running-with-docker)
- [Running Locally](#running-locally)
- [Using the Dashboard](#using-the-dashboard)
- [Experiments](#experiments)
- [Target Performance](#target-performance)
- [Privacy and Research Motivation](#privacy-and-research-motivation)
- [Potential Applications](#potential-applications)
- [Technology Stack](#technology-stack)
- [Future Work](#future-work)
- [Research Direction](#research-direction)

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

# Key Features

- Passive network monitoring using inter-packet arrival times
- No packet payload inspection
- Three-class congestion prediction
- Temporal Convolutional Network for sequence modeling
- Random Forest machine-learning baseline
- Statistical threshold-based baseline
- Synthetic network trace generation
- Configurable congestion scenarios
- Statistical feature extraction
- Model training pipeline
- Model evaluation and comparison
- Confusion matrix and classification metrics
- CPU-based inference
- Flask REST API
- Interactive browser dashboard
- Live congestion simulation
- Early-warning visualization
- Docker-based deployment

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

# Input and Output

## Input

The primary input is a sequence of inter-packet arrival times measured in microseconds.

Example:

```text
[1023, 978, 1102, 995, 1044, 1012, 1087, ...]
```

The sequence is divided into sliding windows before being passed to the prediction models.

The current experimental configuration uses a window of IAT values, while the data generation and experimentation pipeline supports different window sizes.

Example configurations include:

```text
50 IAT values
100 IAT values
150 IAT values
200 IAT values
```

## Output

The system produces a three-class congestion prediction:

```text
0 -> Green
1 -> Yellow
2 -> Red
```

The prediction represents the estimated upcoming network congestion state.

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

The architecture is:

```text
Input IAT Sequence
        |
        v
Input Projection
        |
        v
Residual TCN Blocks
        |
        v
Dilated Causal Convolutions
        |
        v
Temporal Representation
        |
        v
Final Time Step
        |
        v
Linear Classifier
        |
        v
Green / Yellow / Red
```

### TCN Configuration

The current implementation uses:

| Parameter | Value |
|---|---:|
| Input channels | 1 |
| Number of classes | 3 |
| Hidden channels | 64 |
| Kernel size | 7 |
| Residual blocks | 4 |
| Dropout | 0.2 |
| Dilation | Exponentially increasing |

The dilations follow:

```text
1
2
4
8
```

The model uses causal convolutions so that predictions do not depend on future observations.

Residual connections are used between convolutional blocks to support deeper temporal modeling.

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

Typical files include:

```text
saved_models/
├── tcn.pt
├── rf.pkl
├── rule_info.json
├── tcn_history.json
├── train_results.json
└── eval_results.json
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
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix
- Inference latency

Macro-F1 is particularly useful for comparing performance across the three congestion classes.

## Evaluation Workflow

```text
Held-Out Test Set
       |
       +------------------+
       |                  |
       v                  v
Rule Baseline        Random Forest
       |                  |
       +---------+--------+
                 |
                 v
                TCN
                 |
                 v
         Performance Comparison
```

Evaluation results are stored in:

```text
saved_models/eval_results.json
```

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

---

## Data Lab

The Data Lab provides functionality for generating synthetic network traces.

Users can configure traffic scenarios and generate data for model training and testing.

The generated data is stored in the project `data/` directory.

---

## Training Arena

The Training Arena provides an interface for training the available models.

The training workflow includes:

```text
Dataset
   |
   v
Rule Baseline
   |
   v
Random Forest
   |
   v
TCN
   |
   v
Saved Models
```

Training results can be displayed and stored for later evaluation.

---

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

## Live Demo

The Live Demo provides real-time simulation of packet timing and congestion prediction.

Supported scenarios include:

- Normal traffic
- Mild congestion
- Heavy congestion
- Sudden burst events

The dashboard can visualize:

- Packet timing
- Average IAT
- Jitter
- Packet count
- Current congestion state
- Model prediction
- Baseline prediction

---

# Early Warning

A key objective of NetPulse is not only to classify the current congestion state but to provide an early indication of upcoming congestion.

During the live demonstration, the system can track:

- Congestion events detected early
- Events missed by the baseline
- Estimated early-warning time

This allows the prediction model to be evaluated from an operational perspective rather than only using classification accuracy.

---

# Flask API

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

# Project Structure

```text
congestion-predictor/
|
├── backend/
│   |
│   ├── app.py
│   |       Flask API server and application entry point
│   |
│   ├── data_generator.py
│   |       Synthetic network trace generator
│   |
│   ├── features.py
│   |       Statistical feature extraction
│   |
│   ├── train.py
│   |       Training orchestrator
│   |
│   ├── evaluate.py
│   |       Model evaluation and comparison
│   |
│   ├── models/
│   |   |
│   |   ├── tcn.py
│   |   |       Temporal Convolutional Network
│   |   |
│   |   ├── baseline_rf.py
│   |   |       Random Forest baseline
│   |   |
│   |   └── baseline_rule.py
│   |           Rule-based baseline
│   |
│   └── requirements.txt
│
├── frontend/
│   |
│   ├── index.html
│   |       Dashboard interface
│   |
│   ├── style.css
│   |       Dashboard styling
│   |
│   └── app.js
│           Dashboard logic and API communication
│
├── data/
│   |
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   ├── y_test.npy
│   └── metadata.json
│
├── saved_models/
│   |
│   ├── tcn.pt
│   ├── rf.pkl
│   ├── rule_info.json
│   ├── tcn_history.json
│   ├── train_results.json
│   └── eval_results.json
│
├── Dockerfile
├── .dockerignore
└── README.md
```

---

# Installation

NetPulse can be run using Docker or a local Python environment.

Docker is recommended because it provides an isolated environment containing the project's Python and machine-learning dependencies.

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

# Running Locally

## Requirements

Recommended environment:

```text
Python 3.10
```

## Create a Virtual Environment

From the project root:

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Activate on Linux or macOS:

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## Start the Backend

From the `backend` directory:

```bash
python app.py
```

The server will be available at:

```text
http://localhost:5000
```

Open this URL in a browser.

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

If trained model files already exist in `saved_models/`, the application can use those saved checkpoints without retraining.

---

# Training from Scratch

To train the models manually, navigate to the backend directory:

```bash
cd backend
```

Then run:

```bash
python train.py
```

The training pipeline will:

```text
Load / Generate Dataset
        |
        v
Train Rule Baseline
        |
        v
Train Random Forest
        |
        v
Train TCN
        |
        v
Save Models
```

The trained models are saved in:

```text
saved_models/
```

---

# Evaluation from Scratch

After training the models, run:

```bash
python evaluate.py
```

The evaluation pipeline loads the saved models and evaluates them on the test dataset.

The results are stored in:

```text
saved_models/eval_results.json
```

---

# Experiments

NetPulse supports several experiments for studying the relationship between temporal context, model complexity, data volume, and prediction performance.

## 1. Window Size

Test different IAT window sizes:

```text
50
100
150
200
```

Measure:

- Macro-F1
- Precision
- Recall
- Inference latency
- Early-warning capability

The objective is to study the trade-off between temporal context and prediction latency.

---

## 2. TCN Depth

Increase the number of TCN residual blocks:

```text
4
5
6
7
```

Study whether additional temporal depth improves performance.

The experiment can also investigate whether deeper architectures introduce unnecessary computational overhead.

---

## 3. Dataset Size

Generate different amounts of training data:

```text
50 traces
100 traces
200 traces
500 traces
```

Measure the resulting learning curves.

This can be used to determine how much training data is required for stable performance.

---

## 4. Noise and Jitter

Modify the synthetic data generator parameters to introduce different levels of timing noise and jitter.

Evaluate:

- Accuracy degradation
- Robustness
- False alarms
- Missed congestion events

---

## 5. Congestion Transitions

Experiment with different congestion-state transition probabilities.

This allows evaluation of how well the model handles:

```text
Green -> Yellow
Yellow -> Red
Red -> Yellow
Yellow -> Green
```

rather than only isolated congestion events.

---

## 6. Real Network Traces

The synthetic dataset provides a controlled experimental environment.

For external validation, the system can be extended to real network timing traces such as CAIDA or MAWI datasets when appropriate timing information and labels are available.

The real-data experiment should evaluate whether models trained on synthetic data generalize to real network traffic.

---

# Target Performance

The current project defines the following target success criteria:

| Metric | Target |
|---|---:|
| TCN Macro-F1 | >= 0.80 |
| TCN improvement over RF | >= 8 F1 points |
| CPU inference latency | <= 5 ms |
| False-alarm rate | <= 15% |

These values represent project targets and should not be interpreted as achieved results unless confirmed by the evaluation output.

Actual experimental results should be reported from the generated evaluation files.

---

# Privacy and Research Motivation

Traditional network monitoring may rely on packet contents, application-level information, or active probing.

NetPulse investigates whether packet timing information alone can provide useful information about upcoming congestion.

The approach can be represented as:

```text
Packet Payload
      |
      X
   Not Used

Application Content
      |
      X
   Not Used

Active Probing
      |
      X
   Not Required

Packet Timing
      |
      v
Inter-Packet Arrival Times
      |
      v
Temporal Analysis
      |
      v
Congestion Prediction
```

The central research question is:

> Can temporal patterns in packet arrival times support early network congestion prediction without inspecting packet payloads?

This provides a potential approach for environments where packet payloads are encrypted or inaccessible.

However, claims about generalization to specific protocols such as TLS, QUIC, or WireGuard should be validated experimentally using appropriate real-world datasets.

---

# Potential Applications

If validated on real-world traffic, passive congestion prediction could support applications such as:

## Adaptive Video Streaming

Predict upcoming congestion and allow streaming systems to adjust bitrate before severe buffering occurs.

## Cloud Gaming

Provide early warning of network degradation that could affect latency-sensitive gameplay.

## CDN Traffic Engineering

Use congestion forecasts to assist traffic-routing and resource-allocation decisions.

## VoIP and Video Conferencing

Detect emerging network degradation before communication quality becomes unacceptable.

## Edge Computing

Use predicted network conditions to assist workload placement and resource allocation.

## Network Monitoring

Provide passive early-warning information without requiring packet payload inspection.

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

# Design Principles

NetPulse follows several design principles.

## Passive Observation

The system relies on packet timing rather than active probing.

## Payload Independence

The prediction pipeline does not require access to packet payload contents.

## Temporal Modeling

The TCN learns temporal dependencies rather than treating individual packets as independent observations.

## Baseline Comparison

The system includes both a classical machine-learning model and a rule-based model to provide meaningful comparisons against the TCN.

## Reproducible Experiments

Synthetic data generation, training, evaluation, and model storage are organized into reproducible pipeline stages.

## Containerized Deployment

Docker provides a controlled environment for running the application and its machine-learning dependencies.

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

---

# Future Work

Future extensions include:

1. Evaluate the system on real network traces.
2. Study cross-protocol generalization.
3. Evaluate performance across different network environments.
4. Investigate online and continual learning.
5. Evaluate robustness to packet loss.
6. Evaluate robustness to timing noise.
7. Compare TCN with LSTM and GRU models.
8. Compare TCN with Transformer-based architectures.
9. Investigate model calibration and prediction uncertainty.
10. Investigate domain adaptation from synthetic to real network traffic.
11. Evaluate different congestion-labeling strategies.
12. Study adaptive congestion mitigation based on prediction results.
13. Investigate lightweight architectures for edge deployment.
14. Evaluate inference performance under resource constraints.

---

# Research Direction

NetPulse focuses on the following research hypothesis:

> Packet timing contains temporal information about network conditions that can be used for early congestion prediction without inspecting packet payloads.

The project combines:

```text
Passive Network Observation
            +
Inter-Packet Timing Analysis
            +
Temporal Deep Learning
            +
Classical Machine Learning
            +
Statistical Baselines
            +
Early Congestion Prediction
```

The primary objective is to determine whether packet timing patterns contain sufficient predictive information to identify impending congestion before significant network-quality degradation occurs.

---

# Quick Start

## Docker

From the project root:

```bash
docker build -t netpulse .
docker run --rm -p 5000:5000 netpulse
```

Open:

```text
http://localhost:5000
```

## Local Python

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

---

# Summary

NetPulse is a passive network congestion prediction system that uses inter-packet arrival times to forecast upcoming congestion.

The system compares three approaches:

```text
TCN
 |
 +-- Temporal sequence modeling

Random Forest
 |
 +-- Statistical feature-based machine learning

Rule Baseline
 |
 +-- IAT standard-deviation thresholds
```

The project provides an end-to-end workflow covering:

```text
Data Generation
      |
      v
Feature Engineering
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
Real-Time Prediction
      |
      v
Interactive Dashboard
```

The main research objective is to investigate whether passive packet-timing information can provide useful early congestion predictions without requiring access to packet payloads.