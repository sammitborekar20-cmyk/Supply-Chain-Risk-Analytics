# Supply Chain Risk Analytics Using Big Data

A runnable college project for supply-chain risk analysis, shipment-delay prediction, inventory risk monitoring, and a Streamlit dashboard.

## What is included
- Synthetic dataset generator (creates realistic demo data; no external download required).
- Data-processing and ML pipeline using pandas + scikit-learn.
- Optional PySpark processing script.
- Hive SQL examples.
- Optional HDFS upload script.
- Streamlit dashboard.
- Outputs include shipment risk scores, supplier summaries, route summaries, and inventory risk.

> Transparency: the default pipeline runs locally and uses synthetic demo data. It does not require a Hadoop cluster. HDFS/Hive integration requires those services to be installed and configured separately. Synthetic results are for demonstration, not real-world operational decisions.

## 1. Setup (Mac / VS Code terminal)
```bash
cd supply_chain_risk_analytics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run everything (dataset + model + outputs)
```bash
python run_pipeline.py
```

This creates:
- `data/generated/suppliers.csv`
- `data/generated/shipments.csv`
- `data/generated/inventory.csv`
- `outputs/shipment_risk_predictions.csv`
- `outputs/supplier_risk_summary.csv`
- `outputs/route_risk_summary.csv`
- `outputs/inventory_risk_summary.csv`
- `outputs/model_metrics.json`
- `models/shipment_delay_model.joblib`

## 3. Start the dashboard
```bash
streamlit run app.py
```
Open the local URL Streamlit prints, usually http://localhost:8501.

## 4. Optional PySpark processing
Install Java first if needed, then:
```bash
pip install -r requirements-bigdata.txt
python src/spark_processing.py
```
PySpark is optional because Java/Spark setup can be time-consuming on a deadline. The main pipeline works without it.

## 5. Optional HDFS upload
Make sure Hadoop is running and `hdfs dfs -ls /` works, then:
```bash
python src/hdfs_upload.py
```
The script uploads generated CSVs to `/supply_chain_risk/data`.

## 6. Hive queries
Use `sql/hive_analysis.sql` as a demonstration/query script in Hive. It assumes external tables are created over CSV files accessible to Hive. Update paths for your machine/cluster.

## Methodology
1. Generate/ingest supplier, shipment, and inventory data.
2. Validate and clean fields.
3. Engineer features such as historical supplier delay rate and route delay rate.
4. Train a Random Forest classifier to predict delayed shipments.
5. Produce risk probability and an interpretable weighted risk score.
6. Aggregate supplier, route, and inventory risks.
7. Visualize the outputs in Streamlit.

## Risk score
The demo score combines model-estimated delay probability with normalized shipment and supplier risk indicators. It is a project-defined score, not a certified industry standard.

## Viva note
The current demo generates synthetic data so the full workflow can run without a data-download dependency. For a production deployment, replace the generator with real, permitted supply-chain data and validate the model against actual outcomes.
