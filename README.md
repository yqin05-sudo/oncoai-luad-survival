# OncoAI – TCGA-LUAD Survival Prediction Platform

A Python / Streamlit-based clinical decision support tool for lung adenocarcinoma survival prediction.

## Features

- Cox Proportional Hazards Model
- Kaplan-Meier Survival Analysis
- Forest Plot Visualization
- Individual Risk Prediction
- Batch Prediction (Cohort Mode)
- Downloadable Results (CSV / PNG)

## Dataset

- TCGA-LUAD cohort
- Sample size ~487 patients

## Input Variables

- Age
- Gender
- Smoking history
- Tumor stage
- Surgery
- Chemotherapy
- Radiotherapy

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
