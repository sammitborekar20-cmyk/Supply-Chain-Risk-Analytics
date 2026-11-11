
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel


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

OUTPUT_DIR = BASE_DIR / "outputs" / "feature_target_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_OUTPUT = OUTPUT_DIR / "numeric_relationships.csv"
CATEGORICAL_OUTPUT = OUTPUT_DIR / "categorical_relationships.csv"


# --------------------------------------------------
# 2. FEATURE DEFINITIONS
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

TARGET = "is_delayed"

REQUIRED_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + [TARGET]
)


# --------------------------------------------------
# 3. CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("FeatureTargetRelationshipAnalysis")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:

    # --------------------------------------------------
    # 4. LOAD DATASET
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("SUPPLY CHAIN RISK ANALYTICS")
    print("FEATURE-TO-TARGET RELATIONSHIP ANALYSIS")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("\nLoading complete processed dataset...")

    df = spark.read.parquet(str(DATA_PATH))

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select only the required columns.
    df = df.select(*REQUIRED_COLUMNS)

    # Exclude null and invalid target records.
    df = df.filter(
        F.col(TARGET).isin(0, 1)
    )

    # Persist to disk to reduce repeated dataset reads.
    df = df.persist(StorageLevel.DISK_ONLY)

    total_records = df.count()

    print(
        f"\nTotal records analysed: {total_records:,}"
    )


    # --------------------------------------------------
    # 5. TARGET DISTRIBUTION
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    target_distribution = (
        df.groupBy(TARGET)
        .agg(F.count("*").alias("records"))
        .orderBy(TARGET)
    )

    target_distribution.show(truncate=False)


    # --------------------------------------------------
    # 6. NUMERICAL FEATURE RELATIONSHIPS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURE RELATIONSHIPS")
    print("=" * 70)

    numeric_results = []

    for feature in NUMERIC_FEATURES:

        print(f"\nAnalysing: {feature}")

        # Pearson correlation with binary target.
        correlation = df.stat.corr(
            feature,
            TARGET
        )

        # Compare feature statistics between delayed
        # and non-delayed shipments.
        grouped = (
            df.groupBy(TARGET)
            .agg(
                F.count("*").alias("records"),
                F.avg(feature).alias("mean"),
                F.stddev(feature).alias("stddev")
            )
            .orderBy(TARGET)
            .collect()
        )

        result = {
            "feature": feature,
            "pearson_correlation_with_delay": (
                float(correlation)
                if correlation is not None
                else None
            )
        }

        for row in grouped:

            label = int(row[TARGET])

            prefix = (
                "delayed"
                if label == 1
                else "not_delayed"
            )

            result[f"{prefix}_records"] = int(
                row["records"]
            )

            result[f"{prefix}_mean"] = (
                float(row["mean"])
                if row["mean"] is not None
                else None
            )

            result[f"{prefix}_stddev"] = (
                float(row["stddev"])
                if row["stddev"] is not None
                else None
            )

        numeric_results.append(result)

        print(
            "Pearson correlation with is_delayed:",
            correlation
        )

        for row in grouped:

            label_name = (
                "Delayed"
                if row[TARGET] == 1
                else "Not delayed"
            )

            print(
                f"{label_name}: "
                f"records={row['records']:,}, "
                f"mean={row['mean']}, "
                f"stddev={row['stddev']}"
            )


    # --------------------------------------------------
    # 7. SAVE NUMERICAL RESULTS
    # --------------------------------------------------

    numeric_df = spark.createDataFrame(numeric_results)

    numeric_df.coalesce(1).write.mode(
        "overwrite"
    ).option(
        "header", True
    ).csv(
        str(OUTPUT_DIR / "numeric_relationships")
    )

    print(
        "\nNumerical relationship results saved."
    )


    # --------------------------------------------------
    # 8. CATEGORICAL FEATURE RELATIONSHIPS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CATEGORICAL FEATURE RELATIONSHIPS")
    print("=" * 70)

    categorical_results = []

    for feature in CATEGORICAL_FEATURES:

        print(f"\nAnalysing: {feature}")

        # Calculate full-dataset shipment counts and
        # delay rates for every category.
        grouped = (
            df.groupBy(feature)
            .agg(
                F.count("*").alias("total_shipments"),

                F.sum(
                    F.when(
                        F.col(TARGET) == 1, 1
                    ).otherwise(0)
                ).alias("delayed_shipments"),

                F.avg(
                    F.col(TARGET).cast("double")
                ).alias("delay_rate")
            )
            .orderBy(F.desc("delay_rate"))
        )

        rows = grouped.collect()

        print(
            f"\nDelay rate by {feature}:"
        )

        for row in rows:

            delay_rate = (
                float(row["delay_rate"]) * 100
            )

            delayed_count = int(
                row["delayed_shipments"]
            )

            total_count = int(
                row["total_shipments"]
            )

            print(
                f"{row[feature]}: "
                f"{delay_rate:.2f}% delayed "
                f"({delayed_count:,} / "
                f"{total_count:,} shipments)"
            )

            categorical_results.append({
                "feature": feature,
                "category": str(row[feature]),
                "total_shipments": total_count,
                "delayed_shipments": delayed_count,
                "delay_rate_percentage": delay_rate
            })


    # --------------------------------------------------
    # 9. SAVE CATEGORICAL RESULTS
    # --------------------------------------------------

    categorical_df = spark.createDataFrame(
        categorical_results
    )

    categorical_df.coalesce(1).write.mode(
        "overwrite"
    ).option(
        "header", True
    ).csv(
        str(OUTPUT_DIR / "categorical_relationships")
    )

    print(
        "\nCategorical relationship results saved."
    )


    # --------------------------------------------------
    # 10. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nTotal records analysed: {total_records:,}"
    )

    print(
        "\nNumerical results:"
        f"\n{OUTPUT_DIR / 'numeric_relationships'}"
    )

    print(
        "\nCategorical results:"
        f"\n{OUTPUT_DIR / 'categorical_relationships'}"
    )

    print(
        "\nExisting trained model was not modified."
    )

    print(
        "No model retraining was performed."
    )

finally:

    df.unpersist() if "df" in locals() else None

    spark.stop()

    print("\nSpark session stopped.")
