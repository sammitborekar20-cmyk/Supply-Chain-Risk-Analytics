
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

# Use the existing V2 Parquet dataset.
DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "large_dataset_v2"
    / "processed_shipments_parquet"
)

# Save V3 separately. Do not overwrite V1 or V2.
OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "full_dataset_spark_ml_v3_onehot"
)

MODEL_PATH = OUTPUT_DIR / "supply_chain_rf_v3"
METRICS_PATH = OUTPUT_DIR / "full_dataset_metrics_v3.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"V2 dataset not found: {DATA_PATH}"
    )

if MODEL_PATH.exists() or METRICS_PATH.exists():
    raise FileExistsError(
        "V3 output already exists. "
        "Choose a new output directory before training."
    )


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

NUM_PARTITIONS = 16

NUMERIC_COLS = [
    "distance_km",
    "inventory_level",
    "demand_units",
    "expected_days",
    "defective_product"
]

CATEGORICAL_COLS = [
    "traffic_level",
    "weather_condition",
    "transport_mode"
]

LABEL_COL = "is_delayed"


# --------------------------------------------------
# 3. CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SupplyChainFullDatasetTrainingV3")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config(
        "spark.sql.files.maxPartitionBytes",
        "67108864"
    )
    .config(
        "spark.sql.adaptive.enabled",
        "true"
    )
    .config(
        "spark.memory.fraction",
        "0.5"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nSpark version:", spark.version)
print("Input:", DATA_PATH)
print("Output:", OUTPUT_DIR)


# Initialize variables for safe cleanup.
df = None
train_df = None
test_df = None
predictions = None


try:

    # --------------------------------------------------
    # 4. LOAD AND PREPARE DATA
    # --------------------------------------------------

    print("\nLoading processed V2 shipment data...")

    df = spark.read.parquet(str(DATA_PATH))

    required_columns = (
        NUMERIC_COLS
        + CATEGORICAL_COLS
        + [LABEL_COL]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.select(*required_columns)

    # Convert numerical columns to double.
    for column in NUMERIC_COLS + [LABEL_COL]:
        df = df.withColumn(
            column,
            F.col(column).cast("double")
        )

    # Remove records containing null values.
    df = df.dropna(subset=required_columns)

    # Retain valid binary target labels.
    df = df.filter(
        F.col(LABEL_COL).isin(0.0, 1.0)
    )

    print("\nPreparing Spark partitions...")

    df = df.repartition(NUM_PARTITIONS).persist(
        StorageLevel.DISK_ONLY
    )

    total_rows = df.count()

    print(f"\nTotal usable records: {total_rows:,}")

    print("\nLabel distribution:")

    df.groupBy(LABEL_COL) \
        .count() \
        .orderBy(LABEL_COL) \
        .show()

    # --------------------------------------------------
    # 5. TRAIN-TEST SPLIT
    # --------------------------------------------------

    print("\nCreating 80:20 train-test split...")

    train_df, test_df = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

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

    # Release original cached dataframe.
    df.unpersist()
    df = None

    # --------------------------------------------------
    # 6. CALCULATE CLASS WEIGHTS
    # --------------------------------------------------

    print("\nCalculating class weights...")

    class_distribution = (
        train_df
        .groupBy(LABEL_COL)
        .count()
        .collect()
    )

    class_counts = {
        int(row[LABEL_COL]): int(row["count"])
        for row in class_distribution
    }

    if 0 not in class_counts or 1 not in class_counts:
        raise ValueError(
            "Both classes must be present in training data."
        )

    total_training = sum(class_counts.values())

    class_weights = {
        label: total_training / (2.0 * count)
        for label, count in class_counts.items()
    }

    print("Class counts:", class_counts)
    print("Class weights:", class_weights)

    train_df = train_df.withColumn(
        "class_weight",
        F.when(
            F.col(LABEL_COL) == 0.0,
            F.lit(class_weights[0])
        ).otherwise(
            F.lit(class_weights[1])
        )
    )

    # --------------------------------------------------
    # 7. BUILD RANDOM FOREST PIPELINE - V3
    # --------------------------------------------------

    print("\nBuilding V3 OneHotEncoder Random Forest pipeline...")

    # Step 1: Convert categorical text into numeric indices.
    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_index",
            handleInvalid="keep"
        )
        for column in CATEGORICAL_COLS
    ]

    # Step 2: Convert category indices into separate
    # binary feature vectors.
    encoder = OneHotEncoder(
        inputCols=[
            f"{column}_index"
            for column in CATEGORICAL_COLS
        ],
        outputCols=[
            f"{column}_ohe"
            for column in CATEGORICAL_COLS
        ],
        handleInvalid="keep",
        dropLast=False
    )

    # Step 3: Combine numerical and encoded categorical features.
    assembler = VectorAssembler(
        inputCols=(
            NUMERIC_COLS
            + [
                f"{column}_ohe"
                for column in CATEGORICAL_COLS
            ]
        ),
        outputCol="features",
        handleInvalid="keep"
    )

    # Step 4: Random Forest classifier.
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

    # Complete pipeline.
    pipeline = Pipeline(
        stages=indexers + [
            encoder,
            assembler,
            rf
        ]
    )

    # --------------------------------------------------
    # 8. TRAIN MODEL
    # --------------------------------------------------

    print("\nStarting V3 full-dataset model training...")
    print("Categorical encoding: OneHotEncoder")
    print("Keep your Mac plugged in during training.\n")

    start_time = time.time()

    model = pipeline.fit(train_df)

    training_seconds = time.time() - start_time

    print(
        f"\nTraining completed in "
        f"{training_seconds / 60:.2f} minutes."
    )

    # --------------------------------------------------
    # 9. GENERATE TEST PREDICTIONS
    # --------------------------------------------------

    print("\nGenerating test predictions...")

    predictions = model.transform(test_df).persist(
        StorageLevel.DISK_ONLY
    )

    # --------------------------------------------------
    # 10. CONFUSION MATRIX
    # --------------------------------------------------

    matrix_rows = (
        predictions
        .groupBy(LABEL_COL, "prediction")
        .count()
        .collect()
    )

    matrix = {
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "TP": 0
    }

    for row in matrix_rows:

        actual = int(row[LABEL_COL])
        predicted = int(row["prediction"])
        count = int(row["count"])

        if actual == 0 and predicted == 0:
            matrix["TN"] = count

        elif actual == 0 and predicted == 1:
            matrix["FP"] = count

        elif actual == 1 and predicted == 0:
            matrix["FN"] = count

        elif actual == 1 and predicted == 1:
            matrix["TP"] = count

    tn = matrix["TN"]
    fp = matrix["FP"]
    fn = matrix["FN"]
    tp = matrix["TP"]

    # --------------------------------------------------
    # 11. CALCULATE EVALUATION METRICS
    # --------------------------------------------------

    accuracy = (
        (tp + tn) / test_count
        if test_count else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0.0
    )

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

    # --------------------------------------------------
    # 12. SAVE METRICS
    # --------------------------------------------------

    metrics = {
        "model_version": "V3_OneHotEncoder",
        "dataset_version": "V2",
        "total_records": total_rows,
        "training_records": train_count,
        "testing_records": test_count,
        "class_distribution_training": {
            str(key): value
            for key, value in class_counts.items()
        },
        "class_weights": {
            str(key): value
            for key, value in class_weights.items()
        },
        "confusion_matrix": matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "training_time_minutes": training_seconds / 60,
        "model_configuration": {
            "encoding": "StringIndexer + OneHotEncoder",
            "numTrees": 10,
            "maxDepth": 6,
            "maxBins": 16,
            "minInstancesPerNode": 500,
            "subsamplingRate": 0.7,
            "partitions": NUM_PARTITIONS,
            "train_test_split": "80:20",
            "random_seed": 42
        }
    }

    print("\n========== V3 MODEL RESULTS ==========")

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc * 100:.2f}%")

    print("\nConfusion matrix:", matrix)

    # --------------------------------------------------
    # 13. SAVE MODEL AND METRICS
    # --------------------------------------------------

    print("\nSaving V3 trained model...")

    model.write().save(str(MODEL_PATH))

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)

    print("\nV3 TRAINING COMPLETED SUCCESSFULLY.")

finally:

    # --------------------------------------------------
    # 14. CLEANUP
    # --------------------------------------------------

    print("\nReleasing Spark resources...")

    for cached_df in [predictions, train_df, test_df, df]:

        if cached_df is not None:
            try:
                cached_df.unpersist()
            except Exception:
                pass

    spark.stop()

    print("Spark session stopped.")
