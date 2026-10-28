# ANN Lab: Interactive Binary Classification

An educational deep-learning project that trains an Artificial Neural Network (ANN) in PyTorch to classify the two interleaving classes of Scikit-learn's `make_moons` dataset, with an interactive Streamlit dashboard.

## Abstract

A straight line cannot separate the two moon-shaped classes in `make_moons`. This project builds a fully connected ANN with hidden layers and ReLU activations that learns a nonlinear decision boundary. It covers the full workflow: reproducible data generation, leakage-free preprocessing, training with validation monitoring, evaluation on a held-out test set, decision-boundary visualization, architecture comparison, live prediction, and safe model saving and loading.

This is an educational experiment on synthetic data. It is not a real-world diagnostic or fraud-detection system.

## Problem Statement

Given a 2-D point (x1, x2), predict which of two classes (0 or 1) it belongs to. The classes form two interleaving half-circles, so the decision boundary must be nonlinear.

## Objectives

1. Generate a reproducible binary classification dataset with `make_moons`.
2. Implement and train an ANN in PyTorch.
3. Explain layers, weights, biases, activations, loss, and backpropagation.
4. Visualize the learned nonlinear decision boundary.
5. Evaluate on unseen data with accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
6. Compare architectures and hyperparameters on the same data split.
7. Provide an interactive dashboard with live prediction.

## Requirements

**Software**
- Python 3.10 or newer (3.11 recommended)
- PyTorch, Scikit-learn, NumPy, Pandas, Matplotlib, Plotly, Streamlit, Pytest

**Hardware**
- Any modern computer. A CPU is enough, and no GPU is needed.
- About 2 GB of free disk space (mostly PyTorch).

## Architecture and Workflow

```
Generate data -> Split (train / val / test) -> Fit scaler on train only
   -> Train ANN (forward -> loss -> backprop -> Adam step)
   -> Validate each epoch (eval mode, no gradients)
   -> Evaluate once on the test set -> Visualize
```

**Default network**

| Layer | Size | Activation |
|---|---|---|
| Input | 2 features | none |
| Hidden 1 | 16 neurons | ReLU |
| Hidden 2 | 16 neurons | ReLU |
| Output | 1 logit | sigmoid, applied only when producing probabilities |

## Project Structure

```
ann-lab-moons-classifier/
├── app.py               # Streamlit dashboard
├── requirements.txt
├── README.md
├── src/
│   ├── config.py        # defaults, limits, input validation
│   ├── data.py          # dataset generation, splitting, scaling
│   ├── model.py         # PyTorch ANN
│   ├── train.py         # training loop, loss history, seeding
│   ├── evaluate.py      # metrics, confusion matrix
│   ├── visualize.py     # dataset, boundary, loss and confusion plots
│   ├── experiments.py   # fair architecture comparison
│   └── persistence.py   # safe save/load
└── tests/
    ├── test_data.py
    ├── test_model.py
    ├── test_training.py
    └── test_evaluation.py
```

## Installation (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Running

Run the automated tests:

```bash
pytest -q
```

Launch the dashboard:

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Dashboard Pages

| Page | Purpose |
|---|---|
| Overview | Problem, objectives, workflow |
| Dataset Explorer | Sample count, noise and seed controls, scatter plot, class distribution, split sizes |
| ANN Model Lab | Architecture and training controls, Train Model button, progress bar, save/load |
| Decision Boundary | Predicted class regions and optional probability contours |
| Model Evaluation | Metrics, confusion matrix, loss curves |
| Experiment Comparison | Compare architectures and run custom experiments |
| Live Prediction | Choose x1 and x2 with sliders, then predict the class |
| Learn ANN | Short explanations of core concepts |

## Dataset

- **Source:** `sklearn.datasets.make_moons` (synthetic).
- **Features:** `x1`, `x2` are 2-D coordinates with no real-world units.
- **Label:** 0 or 1, one for each moon.
- **Defaults:** 1,000 samples, noise 0.20, seed 42.
- **Split:** 80% train+validation / 20% test (stratified). 20% of the training portion is held out for validation.
- **Scaling:** `StandardScaler` is fitted on the training data only, then applied to validation, test, and live inputs.

## ANN Implementation

- Framework: PyTorch (`nn.Module`, `nn.Linear`, ReLU).
- Loss: `BCEWithLogitsLoss`, which takes raw logits.
- Optimizer: Adam, default learning rate 0.01.
- Defaults: 200 epochs, batch size 32, threshold 0.5.
- Sigmoid is applied only when converting logits to probabilities for prediction and plots.
- All inputs (layers, neurons, learning rate, epochs, batch size, noise, samples, seed) are validated, and invalid values raise clear errors.

## Training Procedure

1. Seed Python, NumPy, and PyTorch for reproducibility.
2. For each epoch, run mini-batches through the forward pass, compute the loss, backpropagate, and update the weights with Adam.
3. Compute the validation loss in `eval()` mode under `torch.no_grad()`, so no weights change.
4. Store the training and validation loss history.
5. Evaluate on the test set only after training finishes.

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| Accuracy | Share of correct predictions |
| Precision | Of points predicted Class 1, the share that truly are Class 1 |
| Recall | Of true Class 1 points, the share the model found |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Chance that a random Class 1 point scores higher than a random Class 0 point |
| Confusion matrix | Counts of TN, FP, FN, TP |

If a class is never predicted, precision and recall are reported as 0 rather than raising an error.

## Results

Fill these in from your own run. Do not use numbers from anywhere else.

**Default configuration** (2 hidden layers x 16 neurons, lr = 0.01, 200 epochs, batch 32):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| ___ | ___ | ___ | ___ | ___ |

**Architecture comparison** (same split and seed):

| Architecture | Accuracy | F1 | ROC-AUC | Train time (s) |
|---|---|---|---|---|
| 1 layer x 8 | ___ | ___ | ___ | ___ |
| 2 layers x 16 | ___ | ___ | ___ | ___ |
| 3 layers x 32 | ___ | ___ | ___ | ___ |

Add screenshots of the decision boundary, loss curves, and confusion matrix here.

## Limitations

- The data is synthetic and low-dimensional, so results do not transfer directly to real-world problems.
- Results depend on the noise level, architecture, learning rate, epochs, and seed. A deeper network is not always better.
- The comparison uses a single seed and a single split. Averaging over several seeds would give more reliable conclusions.
- There is no early stopping or regularization.

## Future Enhancements

- Early stopping, dropout, and weight decay.
- Repeated runs with different seeds, reporting mean and standard deviation.
- Other activation functions (tanh, LeakyReLU) and optimizers (SGD, RMSprop).
- Other datasets (`make_circles`, `make_classification`).
- An animation of the boundary evolving during training.
- Deployment to Streamlit Community Cloud.

## License

MIT (or your choice).
