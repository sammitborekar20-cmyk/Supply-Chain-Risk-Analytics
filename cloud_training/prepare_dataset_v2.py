
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, DoubleType, StringType
)

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "large_dataset_v2" / "*.csv"

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "large_dataset_v2"
    / "processed_shipments_parquet"
)

NUM_PARTITIONS = 16

# Explicit schema avoids expensive CSV schema inference.
schema = StructType([
    StructField("shipment_id", LongType(), True),
    StructField("supplier_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("warehouse_id", IntegerType(), True),
    StructField("route_id", IntegerType(), True),
    StructField("source", StringType(), True),
    StructField("destination", StringType(), True),
    StructField("shipment_date", StringType(), True),
    StructField("transport_mode", StringType(), True),
    StructField("distance_km", IntegerType(), True),
    StructField("traffic_level", StringType(), True),
    StructField("weather_condition", StringType(), True),
    StructField("inventory_level", IntegerType(), True),
    StructField("demand_units", IntegerType(), True),
    StructField("expected_days", IntegerType(), True),
    StructField("actual_days", IntegerType(), True),
    StructField("delivery_delay_days", IntegerType(), True),
    StructField("is_delayed", IntegerType(), True),
    StructField("defective_product", IntegerType(), True),
    StructField("shipping_cost", DoubleType(), True),
    StructField("month", StringType(), True),
    StructField("data_source", StringType(), True)
])

spark = (
    SparkSession.builder
    .appName("SupplyChainDatasetV2Preparation")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "2")
    .config("spark.sql.files.maxPartitionBytes", "67108864")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print("\nReading V2 CSV dataset...")
    print("Input:", CSV_PATH)

    if not CSV_PATH.parent.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {CSV_PATH.parent}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}. "
            "Choose a new output directory rather than overwriting it."
        )

    df = (
        spark.read
        .option("header", True)
        .option("mode", "FAILFAST")
        .schema(schema)
        .csv(str(CSV_PATH))
    )

    print("\nChecking record count...")

    total_rows = df.count()

    print(f"Total records: {total_rows:,}")

    if total_rows != 19_590_000:
        raise ValueError(
            f"Expected 19,590,000 rows, found {total_rows:,}"
        )

    print("\nRepartitioning data...")

    df = df.repartition(NUM_PARTITIONS)

    print("\nWriting Parquet dataset...")

    df.write \
        .mode("errorifexists") \
        .parquet(str(OUTPUT_PATH))

    print("\nV2 PARQUET PREPARATION COMPLETED")
    print("Total rows:", f"{total_rows:,}")
    print("Partitions:", NUM_PARTITIONS)
    print("Output:", OUTPUT_PATH)

finally:
    spark.stop()
