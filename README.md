
# Supply Chain Risk Analytics Using Big Data

A big-data analytics project for shipment delay prediction and supply-chain risk analysis. The project explores large-scale synthetic shipment data, distributed processing with Apache Spark, machine learning, and risk analysis.

## Project Overview

Supply-chain disruptions can lead to delayed deliveries, inventory shortages, and increased operational costs. This project explores how data engineering and machine learning can help identify shipment delay risks from shipment, transport, inventory, and environmental information.

The repository contains two workflows:

1. **Local analytics workflow:** A lightweight pipeline with data generation, machine learning, risk summaries, and a Streamlit dashboard.
2. **Big-data workflow:** A large-scale PySpark pipeline that processes millions of shipment records and trains Random Forest classification models using Spark MLlib.

> **Dataset transparency:** The large-scale datasets used in the experiments are synthetically generated. Reported metrics measure performance on synthetic test data and should not be interpreted as real-world supply-chain prediction accuracy.

## Key Project Highlights

- Generated and worked with **19.59 million synthetic shipment records**.
- Large-scale dataset consists of approximately **2 GB of CSV data**, distributed across 79 files.
- Used PySpark for distributed data processing and Spark MLlib for model training.
- Prepared data in Parquet format for efficient processing.
- Compared three model experiments: V1, V2, and V3.
- Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Explored categorical feature encoding using StringIndexer and OneHotEncoder.

## Best Classification Experiment

The V2 Random Forest model achieved the following results on its held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 67.78% |
| Precision | 52.50% |
| Recall | 68.76% |
| F1-score | 59.54% |
| ROC-AUC | 73.85% |

- Test records: approximately 3.92 million.
- Majority-class baseline accuracy: 65.52%.
- V2 accuracy was 2.26 percentage points above the majority-class baseline.

V2 is the preferred model for default-threshold classification among the three experiments. V3 achieved a slightly higher ROC-AUC but lower accuracy, precision, recall, and F1-score.



## Technology Stack

- **Language:** Python
- **Big Data Processing:** Apache Spark, PySpark
- **Machine Learning:** Spark MLlib, Random Forest
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib / Spark ML model persistence
- **Dashboard:** Streamlit
- **Development:** VS Code, Python virtual environment

## Repository Structure

```text
supply_chain_risk_analytics/
│
├── app.py
├── run_pipeline.py
├── generate_large_supply_chain_dataset.py
├── process_large_dataset.py
├── requirements.txt
├── requirements-bigdata.txt
├── README.md
│
├── cloud_training/
│   ├── prepare_dataset_v2.py
│   ├── validate_dataset_v2.py
│   ├── train_full_dataset.py
│   ├── train_full_dataset_v2.py
│   ├── train_full_dataset_v3.py
│   ├── evaluate_full_dataset.py
│   ├── feature_importance.py
│   ├── analyze_feature_importance_v2.py
│   ├── diagnose_features.py
│   └── feature_target_analysis.py
│
├── docs/
│   ├── architecture.md
│   └── experiments.md
│
├── src/
├── scripts/
├── sql/
├── data/
├── models/
└── outputs/
```

Generated datasets, model artifacts, outputs, backups, and local environment files are excluded from Git using `.gitignore`.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sammitborekar20-cmyk/Supply-Chain-Risk-Analytics.git
cd supply_chain_risk_analytics
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install local workflow dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the local analytics pipeline

```bash
python run_pipeline.py
```

This workflow generates sample supplier, shipment, and inventory data, trains the local model, and creates risk analysis outputs.

### 5. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Big-Data Workflow

The large-scale experiments use PySpark and Spark MLlib.

### Dependencies

Install Java and configure it for your environment. Then install the big-data dependencies:

```bash
pip install -r requirements-bigdata.txt
```

### Dataset preparation

The large-scale workflow uses a synthetic shipment dataset containing 19.59 million records.

The V2 dataset is prepared in Parquet format before model training. The training scripts expect the prepared dataset at the paths configured in their source files.

### Model training

The repository includes separate scripts for the experiments:

```bash
python cloud_training/train_full_dataset.py
python cloud_training/train_full_dataset_v2.py
python cloud_training/train_full_dataset_v3.py
```

For the V2 Parquet preparation and validation workflow:

```bash
python cloud_training/prepare_dataset_v2.py
python cloud_training/validate_dataset_v2.py
```

**Important:** These scripts expect the generated datasets and prepared Parquet files to exist at their configured paths. Review the paths in each script before running them. Full-dataset training can require substantial memory, processing time, and disk space.

For the V3 OneHotEncoder experiment, use the same V2 Parquet dataset to reproduce the encoding comparison.

## Model Experiments

| Experiment | Description | Accuracy | F1-score | ROC-AUC |
|---|---|---:|---:|---:|
| V1 | Initial synthetic dataset and Random Forest pipeline | 73.25% | 81.25% | 85.24% |
| V2 | Improved synthetic dataset and Random Forest | 67.78% | 59.54% | 73.85% |
| V3 | V2 dataset with OneHotEncoder for categorical features | 67.46% | 59.18% | 74.24% |

The V1 and V2 datasets have different target distributions and different underlying synthetic generation logic. Their metrics should therefore not be interpreted as a direct like-for-like model comparison. V2 and V3 use the same V2 dataset and split, making their comparison more meaningful.



## Architecture

The big-data workflow follows this general sequence:

1. Generate synthetic shipment records.
2. Validate dataset structure and label distribution.
3. Prepare the data in Parquet format.
4. Split data into training and test sets.
5. Encode categorical features and assemble model features.
6. Train a Random Forest classifier using Spark MLlib.
7. Evaluate the model on held-out test data.
8. Analyze feature importance and compare experiments.



## Limitations and Future Work

- The large-scale dataset is synthetic, not collected from real logistics operations.
- Test metrics do not establish real-world generalization.
- V2 and V3 have not been validated against real shipment outcomes.
- Threshold tuning and broader model comparisons remain possible improvements.
- Further work can include real-world dataset validation, model explainability, prediction APIs, and deployment.

## Project Disclaimer

This project is developed for academic and experimental purposes. Its predictions and risk scores are not certified operational recommendations and should not be used for actual logistics decisions without validation on appropriate real-world data.
