# Enhanced Legal Predictor

## Overview
The Enhanced Legal Predictor is an AI-powered system to analyze and predict the outcomes of court cases using advanced Natural Language Processing (NLP) with LegalBERT and Machine Learning. It is built for educational, research, and legal analytics purposes, offering automated insights and preliminary judgments on Indian court case scenarios.

**Developers:** NILESH and KESHAV

**Disclaimer:** This tool is for educational and research use only. For actual legal advice or litigation decisions, consult a qualified legal professional.

---

## Features
- **Case Outcome Prediction:** Predicts if a scenario is likely to be 'Guilty' or 'Not Guilty' based on facts.
- **Multi-Algorithm Analysis:** Ensemble of Legal-BERT, Random Forest, and Logistic Regression for robust predictions.
- **IPC Section Detection:** Automatically identifies applicable Indian Penal Code (IPC) sections and relevant acts.
- **Confidence Scoring:** Provides reliability score for each prediction.
- **Legal Guidelines:** Displays likely punishments or penalties for detected sections.
- **Case Report Generation:** Save results and analysis in PDF or DOCX format.
- **User Interface:** Simple Gradio web app for easy interaction.

---

## Usage
1. **Install dependencies:**
   Make sure Python 3.8+ and pip are installed. Required packages include:
    - pandas, numpy, scikit-learn, torch, transformers, datasets, gradio, fpdf, python-docx
   Install via pip:
   ```bash
   pip install pandas numpy scikit-learn torch transformers datasets gradio fpdf python-docx
   ```
2. **Run the web app:**
   ```bash
   python enhanced_legal_predictor.py
   ```
   This will start a local Gradio interface.
3. **Analyze a case:**
    - Enter a detailed legal scenario/text in the provided textbox.
    - Click "Analyze Case". The system will predict the outcome, list detected IPC sections, show applicable punishments, and report confidence.
    - Use "Save as PDF" or "Save as DOCX" to export the analysis report.

---

## Model and Algorithms
- **Legal-BERT Transformer** for advanced understanding of legal language and context.
- **Random Forest** ensemble and **Logistic Regression** (on TF-IDF features) as interpretable ML baselines.
- **Custom keyword matching** for IPC/NDPS/Corruption/Domestic Violence sections, supporting Indian legal context.
- **Weighted voting ensemble** for final decision.

---

## Datasets and Sources
- Realistic, simulated data from Indian Supreme Court, NCRB crime statistics, and major IPC section patterns.
- Supports external CSV input for training.
- **Open legal datasets and literature:**
    - [Case Law Access Project](https://case.law)
    - [Indian Kanoon](https://indiankanoon.org)
    - Chalkidis et al., "Legal-BERT" (arXiv:2010.02559)

---

## Example
**Input:**
> The defendant entered the victim's house at night and stole jewelry worth 50,000 rupees. The victim was threatened with a weapon. Neighbors witnessed the theft and stolen items were recovered from the accused.

**Predicted Output:**
- **Case Outcome:** Guilty (with confidence score)
- **IPC Violations:**
  - Section IPC 378 (Theft)
  - Section IPC 392 (Robbery)
- **Punishments:**
  - Section 378: 3 years imprisonment/fine
  - Section 392: 10 years imprisonment & fine

---

## File Structure
- `enhanced_legal_predictor.py` - Main Python application
- (Accepts legal cases in CSV format or manual text entry)

---

## Limitations
- For research and legal education only, not for use in actual legal proceedings.
- Do not rely solely on predictions – always cross-check with real legal counsel and statutes.
- Primarily trained for Indian Penal Code contexts; adaptation needed for other jurisdictions.

---

## Credits
Developed by NILESH and KESHAV.

## License
This repository is for educational and research purposes only.
