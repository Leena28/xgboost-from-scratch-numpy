# XGBoost Research Paper From Scratch (NumPy Implementation)

A complete implementation of the XGBoost research paper, **XGBoost: A Scalable Tree Boosting System** from scratch using only **NumPy and Python** , including:

* Gradient Boosting Trees
* First & Second Order Optimization
* Exact Greedy Split Finding
* Approximate Split Finding (Weighted Quantile Sketch Inspired)
* Regularized Objective Function
* Missing Value Handling
* Recursive Tree Construction
* Benchmarking against official XGBoost

Built entirely without using sklearn tree implementations.

---

## Features

- XGBoost regression from scratch
- Exact Greedy Algorithm
- Approximate Split Algorithm
- Gradient & Hessian computation
- Regularized leaf weights
- Missing value handling using default directions
- Recursive CART-style tree building
- Benchmarking with official XGBoost
- California Housing dataset support

---

## Project Structure

```bash
xgboost-from-scratch/
│
├── loss.py                  # Gradient & Hessian calculations
├── split.py                 # Exact + Approximate split finding
├── tree.py                  # Recursive tree construction
├── xgboost_scratch.py       # Main XGBoost implementation
├── train.py                 # Training script
├── train_benchmark.py       # Benchmark vs official XGBoost
├── california_housing.csv   # Dataset
└── README.md
```

---

## Core XGBoost Objective

The implementation follows the original XGBoost gain equation:

\text{Gain}=\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}-\gamma

Leaf weights are computed as:

w^*=-\frac{G}{H+\lambda}

Where:

* (G) = Sum of gradients
* (H) = Sum of Hessians
* (\lambda) = L2 regularization
* (\gamma) = Tree complexity penalty

---

# How It Works

## 1. Gradient & Hessian Computation

The model computes first-order and second-order derivatives for squared loss.

Implemented in `loss.py`. 

```python
gradients = 2 * (y_pred - y_true)
hessians = np.full_like(y_true, 2)
```

---

## 2. Split Finding

Implemented in `split.py`. 

### Exact Greedy Algorithm

* Enumerates all possible split points
* Computes gain for each split
* Selects best feature and threshold

### Approximate Algorithm

* Uses weighted quantile style candidate generation
* Faster than exhaustive search
* Inspired by XGBoost’s approximate split method

### Missing Value Handling

Each split evaluates:

* Missing → Left
* Missing → Right

The best default direction is automatically selected.

---

## 3. Tree Construction

Implemented in `tree.py`. 

The algorithm recursively:

1. Finds best split
2. Partitions data
3. Builds left subtree
4. Builds right subtree
5. Creates leaf nodes using optimal leaf weights

---

## 4. Boosting Process

Implemented in `xgboost_scratch.py`. 

Training loop:

1. Initialize predictions using base score
2. Compute gradients & hessians
3. Train regression tree
4. Update predictions
5. Repeat for `n_estimators`

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/xgboost-from-scratch-numpy.git

cd xgboost-from-scratch-numpy
```

Install dependencies:

```bash
pip install numpy pandas xgboost
```

---

# Usage

## Train the Model

Run:

```bash
python train.py
```

Implemented in `train.py`. 

Example:

```python
model = XGBoost(n_estimators=10,learning_rate=0.1,max_depth=3)

model.fit(X, y)

predictions = model.predict(X)
```

---

# Benchmarking

The implementation is benchmarked against the official XGBoost library.

Implemented in:

* `train.py` 
* `train_benchmark.py` 

### Results

| Model            | RMSE   |
| ---------------- | ------ |
| Scratch XGBoost  | 0.7416 |
| Official XGBoost | 0.7226 |

The custom implementation achieves performance close to the official library while being fully implemented from scratch.

---

# Dataset

Uses the California Housing dataset:

```bash
california_housing.csv
```

---

# Hyperparameters

| Parameter       | Description                       |
| --------------- | --------------------------------- |
| `n_estimators`  | Number of boosting rounds         |
| `learning_rate` | Shrinks contribution of each tree |
| `max_depth`     | Maximum tree depth                |
| `lambda`        | L2 regularization                 |
| `gamma`         | Split penalty                     |

---

# Concepts Implemented from the XGBoost Paper

* Gradient Tree Boosting
* Second Order Taylor Expansion
* Regularized Objective Function
* Exact Greedy Split Search
* Approximate Split Search
* Weighted Quantile Sketch Inspired Splits
* Shrinkage (Learning Rate)
* Missing Value Sparsity Awareness

---

# Future Improvements

* Histogram-based split finding
* Parallel split computation
* Classification support
* GPU acceleration
* Early stopping
* Pruning
* Sparse matrix optimization

---

# References

* Original XGBoost Paper:
  *XGBoost: A Scalable Tree Boosting System*
---

# Author

Implemented by **Leena Harpal**
