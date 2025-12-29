Data is not provided due to privacy and stuff idk, the data used is ComputStat Emerging market Data


# Transformer-Based Financial Return Prediction in Emerging Markets

> **A deep learning project exploring whether attention based mechanisms can identify interpretable alpha signals in volatile emerging market equities.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## Project Overview
This project applies a **Transformer Encoder architecture** to predict stock returns in emerging markets using bucketed financial ratios. [cite_start]Unlike traditional "black box" Deep Learning models, this project focuses heavily on **interpretability**—using attention mechanisms to uncover exactly *which* financial signals drive predictions[cite: 1830, 1832].

[cite_start]The model was benchmarked against **XGBoost** and demonstrated competitive performance while offering superior transparency through Attention Rollout and Entropy-based importance analysis[cite: 2017].

## Motivation
[cite_start]Emerging markets are notoriously volatile and inefficient compared to developed markets[cite: 2011]. This project solves two key problems for quantitative finance:
1.  **Non-Linear Patterns:** Captures complex, non linear relationships in financial ratios that linear models miss.
2.  [cite_start]**Trust & Transparency:** Addresses the "Black Box" problem by mapping predictions back to economic drivers (e.g., Momentum, Book-to-Market), making the AI trustable for portfolio managers[cite: 1955, 3014].

## 🛠️ Technical Approach
* [cite_start]**Data Source:** Compustat Global (Emerging Markets, 2006–2025).
* [cite_start]**Preprocessing:** Quantile based discretization (bucketing) of financial ratios to handle outliers and noise.
* [cite_start]**Model Architecture:** * **Custom PyTorch Transformer:** Encoder only configuration (4 layers, 2 heads, embedding dim 64).
* **Optimization:** Hyperparameter tuning via Optuna + SCRUM GPU Cluster (TPE sampler).
* **Validation:** Expanding window approach to simulate real-world trading scenarios.

## Results
The Transformer achieved performance comparable to the industry standard XGBoost baseline while providing deeper insights.

| Metric | XGBoost (Baseline) | **Transformer (My Model)** |
| :--- | :--- | :--- |
| **Accuracy (Test)** | 42.1% | **40.0%** (Competitive) |
| **Class 2 F1 (High Returns)** | 0.32 | **0.41** (Better Recall) |
| **Interpretability** | SHAP only | **Attention Maps + Rollout + SHAP** |

> [cite_start]*Data sourced from Tables 4.1 and 4.7 in the dissertation[cite: 2371, 2513].*

## Interpretability & Insights
Using **Attention Matrix Extraction** and **SHAP**, the model consistently identified economically meaningful signals across different market regimes:

* [cite_start]**Momentum:** Identified as a "global hub" feature in attention rollout[cite: 2845].
* [cite_start]**Value Signals:** Book-to-Market and Earnings-to-Price were consistently top-ranked drivers[cite: 2999].
* **Visual Proof:**
    ![Attention Heatmap](attention_heatmap.png)
    [cite_start]*(Above: Layer-wise attention heatmap showing the model focusing on Momentum and Book-to-Market signals[cite: 2712].)*

## Tech Stack
* **Core:** Python, PyTorch, NumPy, Pandas
* **ML/Stats:** Scikit-learn, XGBoost, Optuna
* **Visualization:** Matplotlib, Seaborn, SHAP
* **DevOps:** Git, VS Code, Jupyter Lab

## How to Run
 **Clone the repository**
    ```bash
    git clone [https://github.com/hs5450/ml-emerging_markets-dissertation.git](https://github.com/hs5450/ml-emerging_markets-dissertation.git)
    cd ml-emerging_markets-dissertation
    ```
```



