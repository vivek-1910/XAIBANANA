import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LogisticGAM, s, f
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load and preprocess the disease_risk.csv dataset
print("Loading dataset...")
df = pd.read_csv('disease_risk.csv')

# Preprocessing: Define features (X) and target (y)
X = df.drop(columns=['disease'])
y = df['disease']

# Split data for evaluation (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a GAM model for disease prediction
# We use s() for continuous variables (age, bmi, systolic_bp)
print("Training Logistic GAM...")
gam = LogisticGAM(s(0) + s(1) + s(2)).fit(X_train, y_train)

# Check accuracy
y_pred = gam.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on test set: {accuracy:.2f}")

# 3. Select one test instance for which the model predicts “disease” (1)
# Find indices where prediction is 1
disease_indices = np.where(y_pred == 1)[0]

if len(disease_indices) == 0:
    print("No instances predicted as 'disease' in the test set. Selecting from training set.")
    y_pred_train = gam.predict(X_train)
    disease_indices = np.where(y_pred_train == 1)[0]
    selected_idx = disease_indices[0]
    patient_data = X_train.iloc[selected_idx]
    actual_label = y_train.iloc[selected_idx]
else:
    selected_idx = disease_indices[0]
    patient_data = X_test.iloc[selected_idx]
    actual_label = y_test.iloc[selected_idx]

print(f"\n--- Selected Patient Details ---")
print(patient_data)
print(f"Actual Label: {'Disease' if actual_label == 1 else 'No Disease'}")
print(f"Predicted Label: Disease")

# 4. Explain the prediction using the intrinsic interpretability of the GAM
# In LogisticGAM: logit(p) = intercept + sum(f_i(x_i))
# We can calculate the contribution of each feature to the log-odds.

# Get the terms (functions f_i applied to the specific patient's values)
# gam.partial_dependence returns the value for each feature
features = X.columns
contributions = []

# gam.partial_dependence(X, term=i, width=None)
# For a single row, we can use the fitted coefficients and basis functions.
# A simpler way in pygam to get the contribution of a specific point:
# The prediction logit(p) is calculated by summing the splines.
# We can use gam.generate_X_grid(X) logic or just evaluate the terms.

# The terms for a specific instance can be extracted using gam.logs['deviance'] etc? No.
# PyGAM makes it easy via `gam.partial_dependence(patient_data_reshaped, term=i)`
patient_reshaped = patient_data.values.reshape(1, -1)

print("\n--- Feature Contributions to Disease Risk (Log-Odds) ---")
for i, col in enumerate(features):
    # partial_dependence returns the contribution f_i(x_i)
    val = gam.partial_dependence(term=i, X=patient_reshaped)[0]
    contributions.append(val)
    impact = "Increases" if val > 0 else "Decreases"
    print(f"{col}: {val:.4f} ({impact} risk)")

intercept = gam.coef_[-1] # Usually the last coefficient is the intercept in pygam?
# Actually, pygam intercept is part of the model but may need to be checked.
# Better way to get intercept and verify sum:
prob = gam.predict_mu(patient_reshaped)[0]
logit_pred = np.log(prob / (1 - prob))
# sum(contributions) + bias = logit_pred
bias = logit_pred - sum(contributions)
print(f"Intercept/Bias: {bias:.4f}")

# 5. Visualize: Feature contribution plot
plt.figure(figsize=(10, 6))
colors = ['salmon' if c > 0 else 'skyblue' for c in contributions]
plt.barh(features, contributions, color=colors)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Contribution to Log-Odds (f(x))')
plt.title(f'Feature Contributions for Selected Patient (Predicted: Disease)')
plt.grid(axis='x', linestyle=':', alpha=0.7)

# Add numeric labels
for i, v in enumerate(contributions):
    plt.text(v, i, f" {v:.2f}", va='center', fontweight='bold' if v > 0 else 'normal')

plt.tight_layout()
plt.savefig('patient_explanation.png')
print("\nVisualization saved as 'patient_explanation.png'")

# (Optional) Global shape plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = features
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    pdep, conf = gam.partial_dependence(term=i, X=XX, width=0.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], conf, c='r', ls='--')
    ax.set_title(titles[i])
    ax.set_xlabel(titles[i])
    ax.set_ylabel('Contribution')

plt.tight_layout()
plt.savefig('global_features.png')
print("Global shape plots saved as 'global_features.png'")

# 6. Provide a brief interpretation
max_contrib_idx = np.argmax(contributions)
min_contrib_idx = np.argmin(contributions)

interpretation = f"""
--- Prediction Explanation ---
The model predicts 'disease' for this patient because the combined effect of their health metrics
results in a positive log-odds score (which corresponds to a probability of {prob:.2%}, which is > 50%).

- Age: {contributions[0]:.4f} (Increases risk)
- BMI: {contributions[1]:.4f} (Increases risk)
- Systolic BP: {contributions[2]:.4f} (Increases risk)

Specifically, the feature '{features[max_contrib_idx]}' is the strongest driver for this prediction,
contributing {contributions[max_contrib_idx]:.2f} to the risk score.
"""

if contributions[min_contrib_idx] < 0:
    interpretation += f"\nOn the other hand, '{features[min_contrib_idx]}' is actually reducing the predicted risk by {-contributions[min_contrib_idx]:.2f}, but not enough to offset the other factors."
else:
    interpretation += "\nAll features for this patient are contributing positively to the disease risk."

print("\n--- Interpretation ---")
print(interpretation)

with open('explanation.txt', 'w') as f:
    f.write(interpretation)
print("\nInterpretation saved as 'explanation.txt'")
