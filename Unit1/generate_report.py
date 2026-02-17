from fpdf import FPDF
import datetime

class PDFReport(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Explainable AI - Banana Problem Report', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
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
    pdf.ln(5)

    # 1. Dataset Overview
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Dataset Overview', 0, 1)
    pdf.set_font('Arial', '', 11)
    overview_text = (
        "The dataset 'disease_risk.csv' contains health metrics for 300 individuals. "
        "It consists of three clinical features: Age, BMI, and Systolic Blood Pressure. "
        "The target variable is 'disease', which indicates the presence (1) or absence (0) of a condition. "
        "The dataset is perfectly balanced with 150 instances for each class."
    )
    pdf.multi_cell(0, 7, overview_text)
    pdf.ln(2)
    
    # 2. Model Performance
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Model Performance', 0, 1)
    pdf.set_font('Arial', '', 11)
    perf_text = (
        "A Generalized Additive Model (GAM) using Logistic regression was trained on 80% of the data. "
        "The model uses smoothing splines (s-terms) for all continuous features to capture non-linear relationships. "
        "\nModel Accuracy on Test Set: 60.00%"
    )
    pdf.multi_cell(0, 7, perf_text)
    pdf.ln(5)

    # 3. Selected Patient Explanation
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Selected Patient Explanation', 0, 1)
    pdf.set_font('Arial', '', 11)
    patient_text = (
        "Patient #266 was selected for analysis as the model correctly predicted 'Disease' for this instance. "
        "\n- Age: 53.0\n- BMI: 29.7\n- Systolic BP: 145.4"
        "\n\nPredicted Probability: 73.48%\nActual Label: Disease"
    )
    pdf.multi_cell(0, 7, patient_text)
    pdf.ln(5)

    # 4. Feature Contribution Analysis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '4. Feature Contribution Analysis', 0, 1)
    pdf.set_font('Arial', '', 11)
    contrib_text = (
        "The contribution of each feature to the log-odds of the prediction was calculated using the partial dependence functions of the GAM: "
        "\n- Age contribution: 0.0778 (Increases risk)"
        "\n- BMI contribution: 0.2180 (Increases risk)"
        "\n- Systolic BP contribution: 0.2353 (Increases risk)"
        "\n- Intercept/Bias: 0.4878"
    )
    pdf.multi_cell(0, 7, contrib_text)
    
    # 5. Graphs
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '5. Visualizations', 0, 1)
    
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 7, 'Local Explanation: Feature Contributions for Patient #266', 0, 1)
    pdf.image('patient_explanation.png', x=10, y=None, w=180)
    pdf.ln(5)
    
    pdf.cell(0, 7, 'Global Explanation: Feature Shape Functions', 0, 1)
    pdf.image('global_features.png', x=10, y=None, w=180)
    pdf.ln(10)

    # 6. Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '6. Summary', 0, 1)
    pdf.set_font('Arial', '', 11)
    summary_text = (
        "The analysis demonstrates that the GAM provides high interpretability by allowing us to see the exact numerical "
        "contribution of each feature. For Patient #266, Systolic BP was the most significant factor pushing the prediction "
        "towards 'Disease'. All features for this specific patient had a positive impact on the risk score, justifying the "
        "model's high-confidence prediction."
    )
    pdf.multi_cell(0, 7, summary_text)

    # Output
    pdf.output('XAI_Banana_Report_PES1UG23AM355.pdf')
    print("PDF Report generated: XAI_Banana_Report_PES1UG23AM355.pdf")

if __name__ == '__main__':
    create_pdf()
