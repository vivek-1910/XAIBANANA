Banana Problem –Unit2 Individual Activity (10 Marks)
Problem Title 1 (For odd SRNs):
Global Feature Effect Exploration using built-in dataset.
Dataset
Use Breast Cancer Dataset from sklearn
Objective :
Experimentally compare PDP vs ALE vs ICE and study feature interaction + correlation
effects.

1. Tasks to be Performed:
1. Load breast cancer dataset from sklearn
2. Train Black Box Model(Random Forest / XGBoost / Neural Network).
3. Select Features for Explanation (select 2 highly important features
(from model importance or SHAP)1 moderately important feature)
4. Generate Explanations(must implement PDP, ICE, ALE(use PyALE library))
5. Perform following Experiments:
a) Test Feature Correlation Effect(Pick 2 correlated features, then compare PDP vs ALE)
b) Subgroup Behaviour(Check ICE curves → Are all samples behaving same?)
c) Model Change Effect(Train second model → Compare PDP shapes)