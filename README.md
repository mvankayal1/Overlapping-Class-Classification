# Overlapping Class Classification using Crisp Decision & Probabilistic Neural Networks

## 📘 Overview

This project studies overlapping class regions in structured tabular data classification using the **Wine Quality dataset**. The goal is to compare **crisp decision-based classification** with **probabilistic neural approaches** like **Probabilistic Neural Networks (PNN)** and a **probabilistic-output interpreted ANN**.

## 🧠 Models Implemented

* **Crisp Decision Network**: Assigns exactly one class per sample using a winner-take-all hard boundary rule.
* **Standard ANN**: Feedforward neural network trained using backpropagation (ReLU + Dense layers), no uncertainty awareness.
* **Probabilistic-Output Neural Model**: Outputs soft likelihood scores interpreted probabilistically, but does **not perform Bayesian posterior weight inference**.
* **Probabilistic Neural Network (PNN)**: Uses distance-based density estimation (Parzen-style kernel approximation) to compute class probabilities and preserve prediction confidence.

## 📊 Results (RMSE Comparison)

| Model                               | RMSE       |
| ----------------------------------- | ---------- |
| Crisp Decision Network              | **0.7607** |
| Probabilistic Neural Network (PNN)  | 3.511      |
| Standard ANN                        | 3.958      |
| Probabilistic-Output Interpreted NN | 5.171      |

**Key Takeaway:** Crisp decisions performed best in our setup due to **consistent normalization and overlap-noise removal**, while **PNN provides stronger theoretical handling of ambiguity** in overlapping feature regions.

## 🍷 Dataset Details

* **Dataset Name:** Wine Quality (Red & White)
* **Original Source:** UCI Machine Learning Repository
* **Dataset Links:**

  * Red wine: `winequality-red.csv`
  * White wine: `winequality-white.csv`
* **Accessed in project using:** TensorFlow Datasets (`tfds.load("wine_quality")`)

### 🧪 Feature Set Includes:

Acidity, pH, Alcohol, Density, Chlorides, Sulphates, Sulfur Dioxide levels, Sugar content, etc.
**Labels:** Discrete wine quality scores (3 to 9), creating natural overlapping feature distributions.

## 🧩 Limitations

* The model labeled “Bayesian” in earlier drafts is **not a true Bayesian Neural Network** (no priors, no posterior sampling).
* Results are **dataset- and preprocessing-dependent**.
* Hyperparameter tuning for kernel sigma in PNN was not performed (future scope).

## ⚙️ Tech Stack

Python, NumPy, Pandas, TensorFlow/Keras, Scikit-Learn, TFDS, Matplotlib

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
jupyter notebook
```

## 👩‍💻 GitHub Upload Status

This repository contains only:

* `/notebooks/*.ipynb`
* `README.md`
* `requirements.txt`
* `.gitignore`

Large datasets and model binaries are intentionally excluded to maintain reproducibility and avoid oversized Git tracking.

## 🗣 Interview-Safe Summary

> *“Crisp decision methods give certainty-driven single-label predictions. PNN computes distance-based density estimates to retain uncertainty-aware confidence, making it theoretically stronger for overlapping class analysis.”*

## 📎 Future Scope

* Kernel sigma optimization for PNN
* Unbiased normalization across all models
* Exploring fully Bayesian posterior-inferred neural approaches

---

⭐ If you want, I can now also help you:

1. generate a **GitHub repo banner image**
2. create a **commit message template**
3. simulate a **project-based interview**

Just say: `NEXT`
