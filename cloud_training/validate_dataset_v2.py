
from pathlib import Path
import glob

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_PATH = "large_dataset_v2/*.csv"

spark = (
    SparkSession.builder
    .appName("SupplyChainDatasetV2Validation")
    .master("local[2]")
    .config("spark.driver.memory", "3g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nLoading V2 dataset...")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)

# 1. Basic dataset information
print("\n========== BASIC INFORMATION ==========")

print("Columns:", len(df.columns))
print("Column names:", df.columns)

print("Total rows:", df.count())

# 2. Check missing values
print("\n========== MISSING VALUES ==========")

null_counts = df.select([
    F.count(
        F.when(F.col(c).isNull(), c)
    ).alias(c)
    for c in df.columns
])

null_counts.show(truncate=False)

# 3. Target distribution
print("\n========== TARGET DISTRIBUTION ==========")

target_distribution = (
    df.groupBy("is_delayed")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") / F.sum("count").over(
                __import__("pyspark").sql.Window.partitionBy()
            ) * 100,
            2
        )
    )
)

target_distribution.show()

# 4. Delay rates by categorical features
print("\n========== CATEGORICAL DELAY RATES ==========")

for column in [
    "traffic_level",
    "weather_condition",
    "transport_mode"
]:

    print(f"\nDelay rate by {column}:")

    (
        df.groupBy(column)
        .agg(
            F.count("*").alias("total_shipments"),
            F.sum("is_delayed").alias("delayed_shipments"),
            F.round(
                F.avg("is_delayed") * 100,
                2
            ).alias("delay_percentage")
        )
        .orderBy("delay_percentage", ascending=False)
        .show()
    )

# 5. Numerical relationships with target
print("\n========== NUMERICAL CORRELATIONS ==========")

numeric_features = [
    "distance_km",
    "inventory_level",
    "demand_units",
    "expected_days",
    "defective_product"
]

for column in numeric_features:

    correlation = df.stat.corr(
        column,
        "is_delayed"
    )

    print(f"{column}: {correlation:.6f}")

# 6. Validate delay logic
print("\n========== DELAY LOGIC VALIDATION ==========")

invalid_rows = df.filter(
    (
        (F.col("delivery_delay_days") < 0)
    )
    |
    (
        F.col("actual_days")
        != F.col("expected_days") + F.col("delivery_delay_days")
    )
    |
    (
        F.col("is_delayed")
        != F.when(
            F.col("delivery_delay_days") > 0, 1
        ).otherwise(0)
    )
)

print("Invalid delay records:", invalid_rows.count())

# 7. Check invalid binary values
print("\n========== LABEL VALIDATION ==========")

invalid_labels = df.filter(
    ~F.col("is_delayed").isin(0, 1)
    |
    ~F.col("defective_product").isin(0, 1)
)

print("Invalid binary labels:", invalid_labels.count())

print("\nV2 DATASET VALIDATION COMPLETED.")

spark.stop()
