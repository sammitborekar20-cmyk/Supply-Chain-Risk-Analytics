
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel, VectorAssembler


# --------------------------------------------------
# 1. PROJECT PATHS
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


# --------------------------------------------------
# 2. FEATURES
# --------------------------------------------------

NUMERIC_FEATURES = [
    "distance_km",
    "inventory_level",
    "demand_units",
    "expected_days",
    "defective_product"
]

CATEGORICAL_FEATURES = [
    "traffic_level",
    "weather_condition",
    "transport_mode"
]


# --------------------------------------------------
# 3. CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SupplyChainFeatureDiagnostics")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:

    # --------------------------------------------------
    # 4. CHECK INPUT PATHS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("SUPPLY CHAIN RISK ANALYTICS")
    print("FEATURE DIAGNOSTIC ANALYSIS")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )


    # --------------------------------------------------
    # 5. LOAD DATASET
    # --------------------------------------------------

    print("\nLoading processed dataset...")

    df = spark.read.parquet(str(DATA_PATH))

    required_features = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    missing_columns = [
        col for col in required_features
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    print("Dataset loaded successfully.")

    print("\nDataset schema:")
    df.select(*required_features).printSchema()


    # --------------------------------------------------
    # 6. CHECK NULL VALUES
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("NULL VALUE ANALYSIS")
    print("=" * 65)

    null_expressions = [
        F.sum(
            F.when(
                F.col(col).isNull(), 1
            ).otherwise(0)
        ).alias(col)
        for col in required_features
    ]

    null_results = df.agg(*null_expressions).first().asDict()

    for feature, null_count in null_results.items():
        print(f"{feature:<25}: {null_count or 0:,} null values")


    # --------------------------------------------------
    # 7. NUMERICAL FEATURE ANALYSIS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("NUMERICAL FEATURE ANALYSIS")
    print("=" * 65)

    numeric_expressions = []

    for col in NUMERIC_FEATURES:

        numeric_expressions.extend([
            F.min(col).alias(f"{col}_min"),
            F.max(col).alias(f"{col}_max"),
            F.avg(col).alias(f"{col}_mean"),
            F.stddev(col).alias(f"{col}_stddev"),
            F.approx_count_distinct(col).alias(
                f"{col}_distinct"
            )
        ])

    numeric_results = (
        df.agg(*numeric_expressions)
        .first()
        .asDict()
    )

    for feature in NUMERIC_FEATURES:

        print(f"\nFeature: {feature}")
        print(f"  Minimum: {numeric_results[f'{feature}_min']}")
        print(f"  Maximum: {numeric_results[f'{feature}_max']}")
        print(f"  Mean: {numeric_results[f'{feature}_mean']}")
        print(f"  Standard deviation: {numeric_results[f'{feature}_stddev']}")
        print(
            "  Approximate distinct values: "
            f"{numeric_results[f'{feature}_distinct']}"
        )


    # --------------------------------------------------
    # 8. CATEGORICAL FEATURE ANALYSIS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 65)

    # Use a small sample for category frequency reporting.
    sample_df = df.sample(
        withReplacement=False,
        fraction=0.01,
        seed=42
    )

    for feature in CATEGORICAL_FEATURES:

        print(f"\nFeature: {feature}")

        sample_df.groupBy(feature).count().orderBy(
            F.desc("count")
        ).show(20, truncate=False)


    # --------------------------------------------------
    # 9. LOAD EXISTING TRAINED MODEL
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("TRAINED MODEL PIPELINE ANALYSIS")
    print("=" * 65)

    model = PipelineModel.load(str(MODEL_PATH))

    print("\nPipeline stages:")

    for index, stage in enumerate(model.stages):

        print(
            f"{index + 1}. "
            f"{stage.__class__.__name__}"
        )


    # --------------------------------------------------
    # 10. CHECK STRING INDEXER MAPPINGS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("CATEGORICAL ENCODING MAPPINGS")
    print("=" * 65)

    for stage in model.stages:

        if isinstance(stage, StringIndexerModel):

            print(
                f"\nInput column: {stage.getInputCol()}"
            )

            print(
                f"Output column: {stage.getOutputCol()}"
            )

            print("Category mappings:")

            for index, label in enumerate(stage.labels):
                print(f"  {label} -> {index}")


    # --------------------------------------------------
    # 11. CHECK VECTOR ASSEMBLER
    # --------------------------------------------------

    assembler = next(
        (
            stage for stage in model.stages
            if isinstance(stage, VectorAssembler)
        ),
        None
    )

    if assembler is None:
        raise ValueError(
            "VectorAssembler not found in model."
        )

    print("\n" + "=" * 65)
    print("FEATURES USED BY RANDOM FOREST")
    print("=" * 65)

    for index, feature in enumerate(
        assembler.getInputCols(),
        start=1
    ):
        print(f"{index}. {feature}")


    # --------------------------------------------------
    # 12. CHECK RANDOM FOREST FEATURE IMPORTANCE
    # --------------------------------------------------

    rf_model = model.stages[-1]

    print("\n" + "=" * 65)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 65)

    importances = rf_model.featureImportances.toArray()

    feature_names = assembler.getInputCols()

    for feature, importance in sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True
    ):

        print(
            f"{feature:<30} "
            f"{importance * 100:>8.2f}%"
        )


    # --------------------------------------------------
    # 13. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("DIAGNOSTIC ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nNo model retraining was performed.")
    print("No existing model files were modified.")

finally:

    spark.stop()

    print("\nSpark session stopped.")
