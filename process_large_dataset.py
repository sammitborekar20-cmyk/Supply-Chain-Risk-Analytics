
# process_large_dataset.py

from pathlib import Path
import json

import joblib
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# 1. PATHS AND DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "large_dataset_2gb"
OUTPUT_DIR = BASE_DIR / "outputs" / "large_dataset"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = OUTPUT_DIR / "processed_shipments_parquet"
MODEL_PATH = MODEL_DIR / "supply_chain_delay_model.joblib"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"

print("=" * 65)
print("SUPPLY CHAIN RISK ASSESSMENT AND PREDICTION")
print("=" * 65)


# ============================================================
# 2. START PYSPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("SupplyChainRiskAnalytics")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:
    # ========================================================
    # 3. LOAD THE LARGE DATASET
    # ========================================================

    csv_pattern = str(DATA_DIR / "shipments_part_*.csv")

    print("\n[1/7] Loading shipment CSV files...")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "DROPMALFORMED")
        .csv(csv_pattern)
    )

    if not df.columns:
        raise RuntimeError(
            f"No CSV columns found. Check the dataset folder: {DATA_DIR}"
        )

    required_columns = [
        "shipment_id",
        "supplier_id",
        "route_id",
        "distance_km",
        "traffic_level",
        "weather_condition",
        "transport_mode",
        "inventory_level",
        "demand_units",
        "expected_days",
        "actual_days",
        "delivery_delay_days",
        "is_delayed",
        "defective_product",
        "shipping_cost",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    row_count = df.count()
    print(f"Rows loaded: {row_count:,}")

    if row_count == 0:
        raise RuntimeError("The dataset is empty.")

    # ========================================================
    # 4. CLEAN AND PREPARE DATA
    # ========================================================

    print("\n[2/7] Cleaning and preparing data...")

    numeric_columns = [
        "distance_km",
        "inventory_level",
        "demand_units",
        "expected_days",
        "actual_days",
        "delivery_delay_days",
        "is_delayed",
        "defective_product",
        "shipping_cost",
    ]

    for column in numeric_columns:
        df = df.withColumn(
            column,
            F.col(column).cast(DoubleType())
        )

    # Remove rows without the target label or essential identifiers.
    df = df.dropna(
        subset=[
            "shipment_id",
            "distance_km",
            "inventory_level",
            "demand_units",
            "expected_days",
            "is_delayed",
        ]
    )

    # Keep valid binary target labels.
    df = df.filter(F.col("is_delayed").isin(0.0, 1.0))

    # Remove duplicate shipment IDs.
    df = df.dropDuplicates(["shipment_id"])

    cleaned_count = df.count()
    print(f"Rows after cleaning: {cleaned_count:,}")

    if cleaned_count < 100:
        raise RuntimeError(
            "Fewer than 100 usable records remain. "
            "Check the CSV files and column values."
        )

    # ========================================================
    # 5. CALCULATE EXPLAINABLE RISK SCORES
    # ========================================================

    print("\n[3/7] Calculating risk scores...")

    # IMPORTANT:
    # The risk score does NOT use is_delayed or actual_days.
    # These fields describe the outcome and must not be used
    # to calculate a risk score intended for prediction.

    # Weather risk: 10 = clear, 50 = rain, 95 = storm.
    df = df.withColumn(
        "weather_risk",
        F.when(
            F.lower(F.col("weather_condition")) == "storm", 95.0
        )
        .when(
            F.lower(F.col("weather_condition")) == "rain", 50.0
        )
        .otherwise(10.0)
    )

    # Traffic risk: 10 = low, 50 = medium, 90 = high.
    df = df.withColumn(
        "traffic_risk",
        F.when(
            F.lower(F.col("traffic_level")) == "high", 90.0
        )
        .when(
            F.lower(F.col("traffic_level")) == "medium", 50.0
        )
        .otherwise(10.0)
    )

    # Inventory risk: compare current stock against demand.
    df = df.withColumn(
        "inventory_risk",
        F.when(
            F.col("inventory_level") < 0.25 * F.col("demand_units"),
            100.0
        )
        .when(
            F.col("inventory_level") < F.col("demand_units"),
            60.0
        )
        .otherwise(20.0)
    )

    # Distance risk: longer routes receive a higher score.
    df = df.withColumn(
        "distance_risk",
        F.least(
            F.col("distance_km") / F.lit(3000.0) * 100.0,
            F.lit(100.0)
        )
    )

    # Environment combines weather and traffic.
    df = df.withColumn(
        "environment_risk",
        F.col("weather_risk") * 0.6
        + F.col("traffic_risk") * 0.4
    )

    # Final score ranges from 0 to 100.
    # Weights: environment 35%, product defects 25%,
    # inventory 25%, and distance 15%.
    df = df.withColumn(
        "risk_score",
        F.least(
            F.greatest(
                F.col("environment_risk") * 0.35
                + F.col("defective_product") * 100.0 * 0.25
                + F.col("inventory_risk") * 0.25
                + F.col("distance_risk") * 0.15,
                F.lit(0.0)
            ),
            F.lit(100.0)
        )
    )

    # Categorize the final score.
    df = df.withColumn(
        "risk_level",
        F.when(F.col("risk_score") <= 30, "Low")
        .when(F.col("risk_score") <= 60, "Medium")
        .otherwise("High")
    )

    # ========================================================
    # 6. SAVE PROCESSED DATA AS PARQUET
    # ========================================================

    print("\n[4/7] Saving processed data in Parquet format...")

    (
        df.write
        .mode("overwrite")
        .parquet(str(PARQUET_PATH))
    )

    print(f"Parquet dataset saved to: {PARQUET_PATH}")

    # ========================================================
    # 7. GENERATE SUPPLIER, ROUTE, AND RISK SUMMARIES
    # ========================================================

    print("\n[5/7] Generating analytics summaries...")

    supplier_summary = (
        df.groupBy("supplier_id")
        .agg(
            F.count("*").alias("shipment_count"),
            F.avg("is_delayed").alias("observed_delay_rate"),
            F.avg("shipping_cost").alias("average_shipping_cost"),
            F.avg("risk_score").alias("average_risk_score"),
        )
        .orderBy(F.desc("average_risk_score"))
        .limit(100)
    )

    supplier_summary.toPandas().to_csv(
        OUTPUT_DIR / "supplier_risk_summary.csv",
        index=False
    )

    route_summary = (
        df.groupBy("route_id")
        .agg(
            F.count("*").alias("shipment_count"),
            F.avg("is_delayed").alias("observed_delay_rate"),
            F.avg("risk_score").alias("average_risk_score"),
        )
        .orderBy(F.desc("average_risk_score"))
        .limit(100)
    )

    route_summary.toPandas().to_csv(
        OUTPUT_DIR / "route_risk_summary.csv",
        index=False
    )

    risk_summary = (
        df.groupBy("risk_level")
        .agg(
            F.count("*").alias("shipment_count"),
            F.avg("risk_score").alias("average_risk_score"),
        )
        .orderBy("risk_level")
    )

    risk_summary.toPandas().to_csv(
        OUTPUT_DIR / "risk_level_summary.csv",
        index=False
    )

    print("Supplier summary saved.")
    print("Route summary saved.")
    print("Risk-level summary saved.")

    # ========================================================
    # 8. PREPARE A SAMPLE FOR MODEL TRAINING
    # ========================================================

    print("\n[6/7] Preparing training data...")

    feature_columns = [
        "distance_km",
        "traffic_level",
        "weather_condition",
        "transport_mode",
        "inventory_level",
        "demand_units",
        "expected_days",
        "defective_product",
    ]

    label_column = "is_delayed"

    # Sample at most 100,000 rows to keep local training
    # manageable on a laptop.
    model_df = (
        df.select(feature_columns + [label_column])
        .dropna(subset=feature_columns + [label_column])
        .limit(100000)
    )

    training_data = model_df.toPandas()

    if len(training_data) < 100:
        raise RuntimeError(
            "Not enough records for training. "
            "Check the dataset and feature columns."
        )

    X = training_data[feature_columns]
    y = training_data[label_column].astype(int)

    if y.nunique() < 2:
        raise RuntimeError(
            "Training data contains only one target class. "
            "A binary classifier requires both delayed and on-time examples."
        )

    numeric_features = [
        "distance_km",
        "inventory_level",
        "demand_units",
        "expected_days",
        "defective_product",
    ]

    categorical_features = [
        "traffic_level",
        "weather_condition",
        "transport_mode",
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")

    # Train model.
    model.fit(X_train, y_train)

    # Predict on held-out test records.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # ========================================================
    # 9. EVALUATE AND SAVE MODEL
    # ========================================================

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    # AUC requires both classes in the test set.
    try:
        auc = roc_auc_score(y_test, probabilities)
    except ValueError:
        auc = None

    metrics = {
        "dataset": "synthetic supply-chain shipment data",
        "rows_loaded_initially": int(row_count),
        "rows_after_cleaning": int(cleaned_count),
        "training_sample_rows": int(len(training_data)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "model": "RandomForestClassifier",
        "risk_score_uses_actual_delay_label": False,
    }

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 65)
    print("MODEL EVALUATION RESULTS")
    print("=" * 65)
    print(json.dumps(metrics, indent=4))

    print("\nFiles created:")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nProcessing completed successfully.")

finally:
    spark.stop()