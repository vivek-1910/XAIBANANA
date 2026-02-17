# Banana Problem – Unit2 Individual Activity
**Total Marks: 10**

---

## Problem Title
**Global Feature Effect Exploration using Built-in Dataset**

### For Odd SRNs

---

## Dataset
- **Source:** Breast Cancer Dataset from `sklearn`

---

## Objective
Experimentally compare **PDP vs ALE vs ICE** and study feature interaction and correlation effects.

---

## Tasks to be Performed

### 1. Load Dataset
- Load breast cancer dataset from sklearn

### 2. Train Black Box Model
- Choose one: Random Forest / XGBoost / Neural Network

### 3. Select Features for Explanation
- **2 highly important features** (from model importance or SHAP)
- **1 moderately important feature**

### 4. Generate Explanations
- Implement **PDP** (Partial Dependence Plot)
- Implement **ICE** (Individual Conditional Expectation)
- Implement **ALE** (Accumulated Local Effects) — use PyALE library

### 5. Perform Experiments

#### a) Test Feature Correlation Effect
- Pick 2 correlated features
- Compare PDP vs ALE analysis for these features

#### b) Subgroup Behaviour
- Check ICE curves
- Analyze: Are all samples behaving the same?

#### c) Model Change Effect
- Train a second model
- Compare PDP shapes between models

---