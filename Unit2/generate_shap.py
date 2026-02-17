import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import os

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_df = pd.DataFrame(X_train_scaled, columns=data.feature_names)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
rf_model.fit(X_train_scaled, y_train)

# SHAP Analysis
explainer = shap.TreeExplainer(rf_model)
shap_results = explainer.shap_values(X_train_scaled)

# Handle different output formats of shap_values
# Sometimes it's a list [neg_vals, pos_vals], sometimes it's (n, m, 2)
if isinstance(shap_results, list):
    shap_values = shap_results[1]
elif isinstance(shap_results, np.ndarray) and len(shap_results.shape) == 3:
    shap_values = shap_results[:, :, 1]
else:
    shap_values = shap_results

# Debugging shapes
print(f"X_train_df shape: {X_train_df.shape}")
print(f"Final shap_values shape: {shap_values.shape}")

# Ensure directory exists
os.makedirs('outputs/visualizations', exist_ok=True)

# Save SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_df, show=False)
plt.savefig('outputs/visualizations/shap_summary.png', bbox_inches='tight', dpi=300)
plt.close()

# Save SHAP Dependence Plots manually
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)
top_features = feature_importance.head(2)['Feature'].tolist()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for i, feat in enumerate(top_features):
    feat_idx = X_train_df.columns.get_loc(feat)
    x_vals = X_train_df.iloc[:, feat_idx].values
    y_vals = shap_values[:, feat_idx]
    
    axes[i].scatter(x_vals, y_vals, alpha=0.5, color='crimson')
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('SHAP Value')
    axes[i].set_title(f'SHAP Dependence: {feat}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/shap_dependence.png', bbox_inches='tight', dpi=300)
plt.close()

print("✓ SHAP visualizations saved to outputs/visualizations/")
