
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "large_dataset_v2"
    / "processed_shipments_parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "full_dataset_spark_ml_v2_dataset_v2"
)
MODEL_PATH = OUTPUT_DIR / "supply_chain_rf_full"
METRICS_PATH = OUTPUT_DIR / "full_dataset_metrics.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if MODEL_PATH.exists() or METRICS_PATH.exists():
    raise FileExistsError(
        "Output already exists. Rename the output folder "
        "before running again to avoid overwriting results."
    )

NUM_PARTITIONS = 16

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
    .appName("SupplyChainFullDatasetTrainingV2")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config("spark.sql.files.maxPartitionBytes", "67108864")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.memory.fraction", "0.5")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nSpark version:", spark.version)
print("Input:", DATA_PATH)
print("Output:", OUTPUT_DIR)


try:
    # --------------------------------------------------
    # 3. LOAD AND PREPARE DATA
    # --------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("\nLoading processed shipment data...")

    df = spark.read.parquet(str(DATA_PATH))

    required_cols = NUMERIC_COLS + CATEGORICAL_COLS + [LABEL_COL]

    missing_cols = [
        col for col in required_cols if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df.select(*required_cols)

    for col in NUMERIC_COLS + [LABEL_COL]:
        df = df.withColumn(col, F.col(col).cast("double"))

    df = df.dropna(subset=required_cols)

    df = df.filter(F.col(LABEL_COL).isin(0.0, 1.0))

    print("\nPreparing Spark partitions...")

    # Shuffle records into manageable processing partitions.
    df = df.repartition(NUM_PARTITIONS).persist(
        StorageLevel.DISK_ONLY
    )

    total_rows = df.count()

    print(f"\nTotal usable records: {total_rows:,}")

    if total_rows < 1000:
        raise ValueError("Not enough records to train the model.")

    print("\nLabel distribution:")
    df.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

    # --------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------

    print("\nCreating 80:20 train-test split...")

    train_df, test_df = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    # Disk-backed persistence avoids holding the full dataset in RAM.
    train_df = train_df.repartition(8).persist(
        StorageLevel.DISK_ONLY
    )

    test_df = test_df.repartition(4).persist(
        StorageLevel.DISK_ONLY
    )

    train_count = train_df.count()
    test_count = test_df.count()

    print(f"Training records: {train_count:,}")
    print(f"Testing records:  {test_count:,}")

    if train_count == 0 or test_count == 0:
        raise ValueError("Train/test split is empty.")

    # The original prepared DataFrame is no longer needed.
    df.unpersist()

    # --------------------------------------------------
    # 5. CLASS WEIGHTS
    # --------------------------------------------------

    print("\nCalculating class weights...")

    class_counts = {
        int(row[LABEL_COL]): row["count"]
        for row in train_df.groupBy(LABEL_COL).count().collect()
    }

    if set(class_counts) != {0, 1}:
        raise ValueError(
            f"Expected both classes in training data: {class_counts}"
        )

    total_train = sum(class_counts.values())

    weight_0 = total_train / (2.0 * class_counts[0])
    weight_1 = total_train / (2.0 * class_counts[1])

    train_df = train_df.withColumn(
        "class_weight",
        F.when(F.col(LABEL_COL) == 0.0, F.lit(weight_0))
        .otherwise(F.lit(weight_1))
    )

    print("Class counts:", class_counts)
    print("Class weights:", {0: weight_0, 1: weight_1})

    # --------------------------------------------------
    # 6. BUILD RANDOM FOREST PIPELINE
    # --------------------------------------------------

    print("\nBuilding Random Forest pipeline...")

    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_index",
            handleInvalid="keep"
        )
        for col in CATEGORICAL_COLS
    ]

    indexed_cols = [
        f"{col}_index" for col in CATEGORICAL_COLS
    ]

    assembler = VectorAssembler(
        inputCols=NUMERIC_COLS + indexed_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    rf = RandomForestClassifier(
        labelCol=LABEL_COL,
        featuresCol="features",
        weightCol="class_weight",
        numTrees=10,
        maxDepth=6,
        maxBins=16,
        minInstancesPerNode=500,
        maxMemoryInMB=256,
        subsamplingRate=0.7,
        featureSubsetStrategy="sqrt",
        seed=42
    )

    pipeline = Pipeline(
        stages=indexers + [assembler, rf]
    )

    # --------------------------------------------------
    # 7. TRAIN ONE MODEL ON THE COMPLETE TRAINING SPLIT
    # --------------------------------------------------

    print("\nStarting full-dataset model training...")
    print("This may take several hours. Keep your Mac plugged in.")

    start_time = time.time()

    model = pipeline.fit(train_df)

    training_seconds = time.time() - start_time

    print(
        f"\nTraining completed in "
        f"{training_seconds / 60:.2f} minutes."
    )

    # --------------------------------------------------
    # 8. EVALUATE ON HELD-OUT TEST DATA
    # --------------------------------------------------

    print("\nGenerating test predictions...")

    predictions = model.transform(test_df)

    confusion = predictions.groupBy(
        LABEL_COL, "prediction"
    ).count().collect()

    matrix = {
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "TP": 0
    }

    for row in confusion:
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

    accuracy = (tp + tn) / test_count
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
        numBins=1000
    )

    roc_auc = evaluator.evaluate(predictions)

    metrics = {
        "total_records": total_rows,
        "training_records": train_count,
        "testing_records": test_count,
        "class_distribution_training": class_counts,
        "confusion_matrix": matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "training_time_minutes": training_seconds / 60,
        "model_configuration": {
            "numTrees": 10,
            "maxDepth": 6,
            "maxBins": 16,
            "minInstancesPerNode": 500,
            "subsamplingRate": 0.7,
            "partitions": NUM_PARTITIONS
        }
    }

    print("\n========== MODEL RESULTS ==========")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc * 100:.2f}%")
    print("\nConfusion matrix:", matrix)

    # --------------------------------------------------
    # 9. SAVE MODEL AND METRICS
    # --------------------------------------------------

    print("\nSaving trained model...")

    model.write().save(str(MODEL_PATH))

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)

    print("\nFULL-DATASET TRAINING COMPLETED SUCCESSFULLY.")

finally:
    # Release disk-backed cached data.
    try:
        train_df.unpersist()
        test_df.unpersist()
    except (NameError, Exception):
        pass

    spark.stop()
