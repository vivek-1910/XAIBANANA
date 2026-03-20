from __future__ import annotations

import datetime
import json
from pathlib import Path

from fpdf import FPDF


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
VISUALS = OUTPUTS / "visualizations"


class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "Explainable AI - Banana Problem Unit 3 Report", 0, 1, "C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")


def load_results() -> dict:
    with open(OUTPUTS / "results.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_pdf() -> None:
    results = load_results()
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Student Details", 0, 1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, "Name: VIVEK GOWDA S", 0, 1)
    pdf.cell(0, 7, "SRN: PES1UG23AM355", 0, 1)
    pdf.cell(0, 7, "College: PES University", 0, 1)
    pdf.cell(0, 7, "Section: F", 0, 1)
    pdf.cell(0, 7, "Subject: Explainable AI", 0, 1)
    pdf.cell(0, 7, "Unit: 3 - Visual Explanations for Banana Classification", 0, 1)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Objective", 0, 1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        0,
        7,
        "For odd SRNs, the assignment requires Task 1 and Task 2 only. "
        "A pretrained ResNet18 model trained on ImageNet was used to study banana class explanations through "
        "vanilla gradient saliency and activation maximization.",
    )
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Input Image and Prediction", 0, 1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        0,
        7,
        f"Input image: {results['image_path']}\n"
        f"Predicted probability for banana class: {results['banana_probability']:.4f}",
    )
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Top Model Predictions", 0, 1)
    pdf.set_font("Arial", "", 11)
    for item in results["top_predictions"]:
        pdf.cell(0, 7, f"{item['label']}: {item['probability']:.4f}", 0, 1)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "4. Task 1 - Vanilla Gradient Saliency", 0, 1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        0,
        7,
        f"Target logit for banana class: {results['saliency_target_logit']:.4f}\n"
        f"Interpretation: {results['task1_interpretation']}",
    )
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "5. Task 2 - Activation Maximization", 0, 1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        0,
        7,
        f"Final optimized banana logit: {results['activation_max_target_logit']:.4f}\n"
        f"Interpretation: {results['task2_interpretation']}",
    )

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "6. Visualizations", 0, 1)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 7, "6.1 Vanilla Gradient Saliency Map", 0, 1)
    pdf.image(str(VISUALS / "task1_saliency_map.png"), x=10, y=None, w=190)
    pdf.ln(3)
    pdf.cell(0, 7, "6.2 Activation Maximization", 0, 1)
    pdf.image(str(VISUALS / "task2_activation_maximization.png"), x=30, y=None, w=150)

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "7. Conclusion", 0, 1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        0,
        7,
        "Vanilla gradients highlighted edge-heavy regions and were visibly noisy, which matches the expected behavior of "
        "gradient-based saliency. Activation maximization produced a yellow curved texture pattern rather than a clean object, "
        "indicating that the CNN encodes banana evidence through a mixture of color and curved-shape cues.",
    )
    pdf.ln(5)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 7, f"Report Generated: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", 0, 1)

    output_path = ROOT / "XAI_Banana_Report_Unit3_PES1UG23AM355.pdf"
    pdf.output(str(output_path))
    print(f"PDF report generated: {output_path.name}")


if __name__ == "__main__":
    create_pdf()
