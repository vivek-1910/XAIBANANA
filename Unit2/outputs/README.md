# Unit 2: Explainable AI Analysis - Output Documentation

## 📁 Project Structure

```
outputs/
├── visualizations/
│   ├── feature_importance.png
│   ├── pdp_plots.png
│   ├── ice_plots.png
│   ├── ale_plots_pyale.png
│   ├── ale_plots_custom.png
│   ├── correlation_effect_analysis.png
│   ├── subgroup_behaviour.png
│   └── model_comparison.png
└── reports/
    └── analysis_report.txt
```

## 📊 Visualizations

### 1. Feature Importance (`feature_importance.png`)
Display of top 10 most important features in the Random Forest model as determined by Mean Decrease Impurity (MDI).
- **Top Features**: worst area (0.154), worst concave points (0.145)

### 2. Partial Dependence Plots (`pdp_plots.png`)
Shows the marginal effect of each feature on predictions, averaging over all other features.
- **Purpose**: Understand global feature-target relationships
- **Features Analyzed**: worst area, worst concave points

### 3. Individual Conditional Expectation (`ice_plots.png`)
Individual curves showing how predictions change for each sample when a feature varies.
- **Purpose**: Detect heterogeneity in model behavior across different samples
- **Key Finding**: High heterogeneity (std dev: 0.34-0.38) indicates varying treatment effects

### 4. ALE Plots - PyALE Library (`ale_plots_pyale.png`)
Accumulated Local Effects computed using the official PyALE library with 95% confidence intervals.
- **Purpose**: Unbiased feature effects handling correlated features better than PDP
- **Library**: PyALE v1.2.0
- **Grid Points**: 21 per feature

### 5. ALE Plots - Custom Implementation (`ale_plots_custom.png`)
Custom ALE implementation for comparison and validation purposes.
- **Purpose**: Alternative approach to ALE calculation
- **Method**: Binned local effect accumulation with 50 bins

### 6. Correlation Effect Analysis (`correlation_effect_analysis.png`)
Comparison of PDP vs ALE for highly correlated features.
- **Correlated Pair**: mean radius ↔ mean perimeter (r=0.9979)
- **Key Insight**: PDP and ALE show different patterns for correlated features; ALE better accounts for data distribution

### 7. Subgroup Behaviour Analysis (`subgroup_behaviour.png`)
Detailed ICE analysis with color-coded curves representing different data regions.
- **Purpose**: Identify subgroup-specific feature relationships
- **Heterogeneity Level**: High (std dev: 0.3439-0.3800)

### 8. Model Comparison (`model_comparison.png`)
PDP comparison between Random Forest and Gradient Boosting models.
- **Model 1**: Random Forest (100 trees, max_depth=15)
- **Model 2**: Gradient Boosting (100 trees, max_depth=5)
- **Test Accuracy**: Both models achieve 96.49%
- **Finding**: Different models exhibit different PDP shapes, suggesting model-dependent feature relationships

## 📄 Reports

### Analysis Report (`reports/analysis_report.txt`)
Comprehensive summary including:
- Dataset characteristics (569 samples, 30 features)
- Model performance metrics
- Experiment findings for each of the three experiments
- Technical conclusions and recommendations
- Key insights for each interpretability method

## 🔍 Key Findings

### Method Comparison
| Method | Strength | Limitation |
|--------|----------|-----------|
| **PDP** | Efficient, easily interpretable | Biased by feature correlation |
| **ICE** | Reveals heterogeneity | Prone to overplotting |
| **ALE** | Handles correlation well | More complex to interpret |

### Experiment Results

1. **Feature Correlation Effect**: ALE provides more reliable interpretation for correlated features
2. **Subgroup Behavior**: Significant heterogeneity detected (std dev > 0.34) indicating diverse model behavior
3. **Model Architecture**: Different ensemble methods produce different feature dependency patterns

## 🛠️ Technical Details

- **Dataset**: Breast Cancer Classification (scikit-learn)
- **Primary Model**: Random Forest (100 estimators)
- **Secondary Model**: Gradient Boosting
- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Training Split**: 80% train, 20% test
- **Dependencies**: scikit-learn, PyALE, matplotlib, pandas, numpy

## 📝 Usage Notes

All visualizations are generated at 300 DPI for publication quality (except PyALE at 100 DPI).
The analysis report provides detailed interpretation of findings and recommendations for practical application.

---
**Analysis Completed**: February 17, 2026  
**Status**: All outputs successfully organized and documented
