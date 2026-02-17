from fpdf import FPDF
import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Explainable AI - Global Feature Effect Analysis Report', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def create_pdf():
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Student Information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Student Details', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 7, 'Name: VIVEK GOWDA S', 0, 1)
    pdf.cell(0, 7, 'SRN: PES1UG23AM355', 0, 1)
    pdf.cell(0, 7, 'College: PES University', 0, 1)
    pdf.cell(0, 7, 'Section: F', 0, 1)
    pdf.cell(0, 7, 'Subject: Explainable AI', 0, 1)
    pdf.cell(0, 7, 'Unit: 2 - Global Feature Effect Exploration', 0, 1)
    pdf.ln(5)

    # 1. Dataset Overview
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Dataset Overview', 0, 1)
    pdf.set_font('Arial', '', 11)
    overview_text = (
        "The Breast Cancer Classification dataset from scikit-learn comprises 569 instances with 30 continuous features. "
        "Features represent various cell nucleus characteristics including radius, texture, perimeter, area, smoothness, compactness, "
        "concavity, concave points, symmetry, and fractal dimension. Each measurement has mean, standard error, and worst value variants. "
        "The binary target variable indicates malignant (1) or benign (0) diagnosis. The dataset was split 80-20 for training and testing."
    )
    pdf.multi_cell(0, 7, overview_text)
    pdf.ln(2)
    
    # 2. Model Performance
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Model Development and Performance', 0, 1)
    pdf.set_font('Arial', '', 11)
    perf_text = (
        "Random Forest Classifier (100 trees, max_depth=15) and Gradient Boosting Classifier were trained on standardized features. "
        "\n\nRandom Forest Performance:\n- Training Accuracy: 100.00%\n- Test Accuracy: 96.49%\n- Top Features: worst area (0.154), worst concave points (0.145)"
        "\n\nGradient Boosting Performance:\n- Training Accuracy: 100.00%\n- Test Accuracy: 96.49%"
    )
    pdf.multi_cell(0, 7, perf_text)
    pdf.ln(2)

    # 3. Explanation Methods
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Explainability Methods Implemented', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '3.1 Partial Dependence Plot (PDP)', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, 
        "Shows the marginal effect of features on predictions after averaging over other features. "
        "PDP is computationally efficient but may be biased by feature correlations."
    )
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '3.2 Individual Conditional Expectation (ICE)', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7,
        "Displays individual-level predictions showing how each sample responds to feature changes. "
        "Reveals heterogeneity in model behavior across different subgroups with std dev: 0.34-0.38."
    )
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '3.3 Accumulated Local Effects (ALE)', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7,
        "Unbiased feature effects computed using PyALE library with 21 grid points and 95% confidence intervals. "
        "Better handles feature correlations compared to PDP."
    )

    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '3.4 SHAP (SHapley Additive exPlanations)', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7,
        "Unified approach to explaining predictions based on game theory. Provides both global importance "
        "(summary plots) and detailed feature impact analysis (dependence plots)."
    )
    pdf.ln(2)

    # 4. Experiment Results
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '4. Experimental Findings', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '4.1 Feature Correlation Effect', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7,
        "Analysis of highly correlated feature pair: mean radius and mean perimeter (r=0.9979). "
        "Key Finding: PDP and ALE exhibit different patterns for correlated features. ALE provides more reliable interpretation "
        "by accounting for feature dependencies in the data distribution."
    )
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '4.2 Subgroup Behavior Analysis', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7,
        "ICE curve analysis reveals high heterogeneity in model predictions. Different samples demonstrate varying sensitivity "
        "to feature changes (std dev: 0.3439-0.3800), indicating diverse decision boundaries across subgroups."
    )
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '4.3 Model Architecture Comparison', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7,
        "PDP comparison between Random Forest and Gradient Boosting shows different curve shapes, "
        "suggesting model-specific feature dependency patterns despite achieving identical test accuracy (96.49%)."
    )
    
    # 5. Key Insights
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '5. Key Insights and Conclusions', 0, 1)
    pdf.set_font('Arial', '', 11)
    insights_text = (
        "1. PDP provides efficient global understanding but may be misleading for correlated features.\n\n"
        "2. ALE plots represent superior choice for models with feature dependencies due to unbiased estimation.\n\n"
        "3. ICE curves reveal critical heterogeneity patterns indicating non-uniform treatment effects across samples.\n\n"
        "4. Feature importance rankings align with PDP magnitude, validating model decision boundaries.\n\n"
        "5. Model architecture selection affects feature dependency patterns; comprehensive analysis requires multi-method comparison."
    )
    pdf.multi_cell(0, 7, insights_text)
    
    # 6. Visualizations
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '6. Visualizations and Analysis', 0, 1)
    pdf.ln(5)
    
    # Feature Importance
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.1 Feature Importance Ranking', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/feature_importance.png', x=10, y=None, w=190)
    pdf.ln(2)
    
    # PDP
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.2 Partial Dependence Plots - Top Features', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/pdp_plots.png', x=10, y=None, w=190)
    pdf.ln(2)
    
    # ICE
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.3 Individual Conditional Expectation Plots', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/ice_plots.png', x=10, y=None, w=190)
    pdf.ln(2)
    
    # ALE PyALE
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.4 Accumulated Local Effects (PyALE Library)', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/ale_plots_pyale.png', x=10, y=None, w=190)
    pdf.ln(2)
    
    # ALE Custom
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.5 Accumulated Local Effects (Custom Implementation)', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/ale_plots_custom.png', x=10, y=None, w=190)
    pdf.ln(2)
    
    # Correlation Analysis
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.6 PDP vs ALE Comparison for Correlated Features', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/correlation_effect_analysis.png', x=10, y=None, w=190)
    pdf.ln(2)
    
    # Subgroup Behavior
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.7 Subgroup Behavior Analysis - ICE Heterogeneity', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/subgroup_behaviour.png', x=10, y=None, w=190)
    pdf.ln(2)
    
    # Model Comparison
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.8 Model Comparison - Random Forest vs Gradient Boosting', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/model_comparison.png', x=10, y=None, w=190)
    pdf.ln(2)

    # SHAP Summary
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.9 SHAP Summary Plot - Feature Impacts', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/shap_summary.png', x=10, y=None, w=190)
    pdf.ln(2)

    # SHAP Dependence
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, '6.10 SHAP Dependence Analysis - Top Features', 0, 1)
    if pdf.will_page_break(60):
        pdf.add_page()
    pdf.image('outputs/visualizations/shap_dependence.png', x=10, y=None, w=190)
    
    # 7. Recommendations
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '7. Recommendations and Future Work', 0, 1)
    pdf.set_font('Arial', '', 11)
    recommendations = (
        "1. Adopt multi-method interpretability approach combining PDP, ICE, and ALE for comprehensive analysis.\n\n"
        "2. Prioritize ALE plots when features exhibit high correlation (r > 0.7) for unbiased interpretation.\n\n"
        "3. Use ICE curves to identify and segment heterogeneous subgroups for targeted model refinement.\n\n"
        "4. Compare PDP shapes across models to validate feature relationships and detect overfitting.\n\n"
        "5. Document interpretability findings for regulatory compliance and stakeholder communication.\n\n"
        "6. Extend analysis to SHAP values and LIME for local explanations of individual predictions."
    )
    pdf.multi_cell(0, 7, recommendations)
    
    pdf.ln(5)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, f'Report Generated: {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}', 0, 1)
    
    pdf.output('XAI_Banana_Report_Unit2_PES1UG23AM355.pdf')
    print("✓ PDF report generated successfully: XAI_Banana_Report_Unit2_PES1UG23AM355.pdf")

if __name__ == '__main__':
    create_pdf()
