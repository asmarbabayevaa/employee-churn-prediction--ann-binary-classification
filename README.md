# Employee Churn Prediction — ANN Binary Classification

A deep learning binary classification project that predicts whether an employee will leave the company (`LeaveOrNot`), using an Artificial Neural Network (ANN) with automated hyperparameter tuning via **Optuna**. Model performance is evaluated using **AUC-ROC** and **Gini coefficient**.

## Requirements

```bash
pip install tensorflow optuna scikit-learn pandas numpy matplotlib seaborn
```

---

## Pipeline Overview

### 1. Import Libraries
Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow`, `optuna`, `sklearn`

### 2. Load Data
Reads `Employee.csv` into a DataFrame and performs an initial inspection with `.describe()` and `.isnull().sum()`.

### 3. Exploratory Data Analysis (EDA)
Boxplots visualized for all continuous numeric columns (excluding `PaymentTier` and `LeaveOrNot`) to inspect distributions and potential outliers.

### 4. Encoding
`pd.get_dummies()` applied to categorical columns with `drop_first=True` to avoid multicollinearity.

### 5. Define Target & Features
- **Target:** `LeaveOrNot` (binary: 0 = stays, 1 = leaves)
- **Features:** All remaining columns after dropping the target

### 6. Scaling
`StandardScaler` fitted on input features and applied via `transform()`. Save the scaler for reuse in deployment or other notebooks.

### 7. Train / Test Split
80/20 split with `random_state=42`.

### 8. Hyperparameter Tuning with Optuna
Optuna searches over:
- Number of units in layer 1 and layer 2
- Optimizer (`adam`, `sgd`, `rmsprop`, `adagrad`)
- Learning rate (log-uniform)
- Epochs and batch size

15 trials, maximizing **AUC-ROC** score on the test set.

### 9. Build Best Model
Reconstructs the ANN using the best hyperparameters found by Optuna.

### 10. Compile Best Model
Compiles with `binary_crossentropy` loss and `AUC` metric using the best optimizer and learning rate.

### 11. Evaluate
Custom `evaluate()` function reports **Gini coefficient** (= 2 × AUC − 1) on both train and test sets in a formatted DataFrame.

---

##  Model Architecture

```
Input Layer
   ↓
Dense(n_units, activation='relu')
   ↓
Dense(n_units, activation='relu')
   ↓
Dense(1, activation='sigmoid')   ← binary classification output
```

---

##  Metrics

| Metric | Description |
|--------|-------------|
| AUC-ROC | Area under the ROC curve — used as Optuna objective (higher = better) |
| Gini | Gini = 2 × AUC − 1 — normalized discriminatory power (higher = better) |

---
