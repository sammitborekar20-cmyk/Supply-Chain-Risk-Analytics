
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "large_dataset_test"
    / "processed_shipments_parquet"
)

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "full_dataset_spark_ml_v2"
    / "supply_chain_rf_full"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "full_dataset_evaluation"

METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
THRESHOLD_PATH = OUTPUT_DIR / "threshold_analysis.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if METRICS_PATH.exists() or THRESHOLD_PATH.exists():
    raise FileExistsError(
        "Evaluation output already exists. "
        "Rename or move the old evaluation folder first."
    )

NUMERIC_COLS = [
    "distance_km",
    "inventory_level",
    "demand_units",
    "expected_days",
    "defective_product",
]

CATEGORICAL_COLS = [
    "traffic_level",
    "weather_condition",
    "transport_mode",
]

LABEL_COL = "is_delayed"


# --------------------------------------------------
# 2. START SPARK
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SupplyChainModelEvaluation")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config("spark.sql.files.maxPartitionBytes", "67108864")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

test_df = None
predictions = None

try:

    # --------------------------------------------------
    # 3. CHECK FILES
    # --------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset missing: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model missing: {MODEL_PATH}")

    print("\nLoading existing trained model...")

    model = PipelineModel.load(str(MODEL_PATH))

    print("Model loaded successfully.")
    print("\nReading processed shipment data...")

    df = spark.read.parquet(str(DATA_PATH))

    required_cols = NUMERIC_COLS + CATEGORICAL_COLS + [LABEL_COL]

    df = df.select(*required_cols)

    for col in NUMERIC_COLS + [LABEL_COL]:
        df = df.withColumn(col, F.col(col).cast("double"))

    df = df.dropna(subset=required_cols)

    df = df.filter(F.col(LABEL_COL).isin(0.0, 1.0))

    # Reproduce the original training script's split preparation.
    df = df.repartition(16).persist(StorageLevel.DISK_ONLY)

    total_rows = df.count()

    print(f"Total usable records: {total_rows:,}")

    # --------------------------------------------------
    # 4. RECREATE THE ORIGINAL TEST SET
    # --------------------------------------------------

    print("\nRecreating original 80:20 split...")

    train_df, test_df = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    # Match the original training script.
    test_df = test_df.repartition(4).persist(
        StorageLevel.DISK_ONLY
    )

    test_count = test_df.count()

    df.unpersist()

    print(f"Test records: {test_count:,}")

    # --------------------------------------------------
    # 5. GENERATE PREDICTIONS
    # --------------------------------------------------

    print("\nGenerating predictions using existing model...")

    predictions = model.transform(test_df).persist(
        StorageLevel.DISK_ONLY
    )

    # Materialize predictions so repeated evaluations can reuse them.
    predictions.count()

    # --------------------------------------------------
    # 6. CONFUSION MATRIX
    # --------------------------------------------------

    print("\nCalculating confusion matrix...")

    rows = predictions.groupBy(
        LABEL_COL, "prediction"
    ).count().collect()

    matrix = {
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "TP": 0
    }

    for row in rows:
        actual = int(row[LABEL_COL])
        predicted = int(row["prediction"])

        if actual == 0 and predicted == 0:
            matrix["TN"] = row["count"]
        elif actual == 0 and predicted == 1:
            matrix["FP"] = row["count"]
        elif actual == 1 and predicted == 0:
            matrix["FN"] = row["count"]
        elif actual == 1 and predicted == 1:
            matrix["TP"] = row["count"]

    tn = matrix["TN"]
    fp = matrix["FP"]
    fn = matrix["FN"]
    tp = matrix["TP"]

    # --------------------------------------------------
    # 7. CLASSIFICATION METRICS
    # --------------------------------------------------

    accuracy = (tp + tn) / test_count

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / (tp + fn) if (tp + fn) else 0.0

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    balanced_accuracy = (recall + specificity) / 2

    # --------------------------------------------------
    # 8. ROC-AUC AND PR-AUC
    # --------------------------------------------------

    print("\nCalculating ROC-AUC and PR-AUC...")

    roc_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
        numBins=1000
    )

    pr_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
        numBins=1000
    )

    roc_auc = roc_evaluator.evaluate(predictions)
    pr_auc = pr_evaluator.evaluate(predictions)

    # --------------------------------------------------
    # 9. MAJORITY CLASS BASELINE
    # --------------------------------------------------

    print("\nCalculating majority-class baseline...")

    label_counts = {
        int(row[LABEL_COL]): row["count"]
        for row in test_df.groupBy(LABEL_COL).count().collect()
    }

    majority_count = max(label_counts.values())
    majority_baseline = majority_count / test_count

    # --------------------------------------------------
    # 10. THRESHOLD ANALYSIS
    # --------------------------------------------------

    print("\nAnalyzing classification thresholds...")

    predictions = predictions.withColumn(
        "positive_probability",
        vector_to_array(F.col("probability")).getItem(1)
    )

    threshold_results = []

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

        predicted_positive = (
            F.col("positive_probability") >= threshold
        )

        actual_positive = F.col(LABEL_COL) == 1.0
        actual_negative = F.col(LABEL_COL) == 0.0

        result = predictions.agg(
            F.sum(
                F.when(actual_negative & ~predicted_positive, 1)
                .otherwise(0)
            ).alias("TN"),

            F.sum(
                F.when(actual_negative & predicted_positive, 1)
                .otherwise(0)
            ).alias("FP"),

            F.sum(
                F.when(actual_positive & ~predicted_positive, 1)
                .otherwise(0)
            ).alias("FN"),

            F.sum(
                F.when(actual_positive & predicted_positive, 1)
                .otherwise(0)
            ).alias("TP")
        ).first()

        t_tn = int(result["TN"] or 0)
        t_fp = int(result["FP"] or 0)
        t_fn = int(result["FN"] or 0)
        t_tp = int(result["TP"] or 0)

        t_precision = (
            t_tp / (t_tp + t_fp)
            if (t_tp + t_fp) else 0.0
        )

        t_recall = (
            t_tp / (t_tp + t_fn)
            if (t_tp + t_fn) else 0.0
        )

        t_f1 = (
            2 * t_precision * t_recall /
            (t_precision + t_recall)
            if (t_precision + t_recall) else 0.0
        )

        threshold_results.append({
            "threshold": threshold,
            "TN": t_tn,
            "FP": t_fp,
            "FN": t_fn,
            "TP": t_tp,
            "precision": t_precision,
            "recall": t_recall,
            "f1_score": t_f1
        })

    # --------------------------------------------------
    # 11. SAVE RESULTS
    # --------------------------------------------------

    metrics = {
        "total_records": total_rows,
        "test_records": test_count,
        "test_class_distribution": label_counts,
        "confusion_matrix": matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "majority_class_baseline": majority_baseline
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    with open(THRESHOLD_PATH, "w") as file:
        json.dump(threshold_results, file, indent=4)

    # --------------------------------------------------
    # 12. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n========== DETAILED EVALUATION ==========")

    print(f"Total records:       {total_rows:,}")
    print(f"Test records:        {test_count:,}")

    print(f"\nAccuracy:             {accuracy * 100:.2f}%")
    print(f"Majority baseline:   {majority_baseline * 100:.2f}%")
    print(f"Precision:            {precision * 100:.2f}%")
    print(f"Recall:               {recall * 100:.2f}%")
    print(f"Specificity:          {specificity * 100:.2f}%")
    print(f"F1 score:             {f1 * 100:.2f}%")
    print(f"Balanced accuracy:   {balanced_accuracy * 100:.2f}%")
    print(f"ROC-AUC:              {roc_auc * 100:.2f}%")
    print(f"PR-AUC:               {pr_auc * 100:.2f}%")

    print("\nConfusion matrix:", matrix)

    print("\nThreshold analysis:")

    for item in threshold_results:
        print(
            f"Threshold: {item['threshold']:.1f} | "
            f"Precision: {item['precision'] * 100:.2f}% | "
            f"Recall: {item['recall'] * 100:.2f}% | "
            f"F1: {item['f1_score'] * 100:.2f}%"
        )

    print("\nMetrics saved:", METRICS_PATH)
    print("Threshold results saved:", THRESHOLD_PATH)

    print("\nEVALUATION COMPLETED SUCCESSFULLY.")

finally:

    if predictions is not None:
        predictions.unpersist()

    if test_df is not None:
        test_df.unpersist()

    spark.stop()
