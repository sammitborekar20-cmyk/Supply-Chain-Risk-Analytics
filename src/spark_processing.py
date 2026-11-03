"""Optional PySpark processing demo. Requires Java and pyspark to be installed."""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
OUT = ROOT / "outputs" / "spark"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.appName("SupplyChainRiskAnalytics")
             .master("local[*]").getOrCreate())
    shipments = spark.read.option("header", True).option("inferSchema", True).csv(str(DATA / "shipments.csv"))
    suppliers = spark.read.option("header", True).option("inferSchema", True).csv(str(DATA / "suppliers.csv"))
    shipments = shipments.withColumn("is_delayed", F.col("is_delayed").cast("int"))
    print("Shipment rows:", shipments.count())
    supplier_summary = (shipments.groupBy("supplier_id")
        .agg(F.count("*").alias("shipment_count"),
             F.round(F.avg("is_delayed")*100, 2).alias("delay_rate_pct"),
             F.round(F.avg("delivery_delay_days"), 2).alias("avg_delay_days")))
    supplier_summary.join(suppliers, "supplier_id", "left").orderBy(F.desc("delay_rate_pct")) \
        .write.mode("overwrite").option("header", True).csv(str(OUT / "supplier_summary"))
    route_summary = (shipments.groupBy("route_id")
        .agg(F.count("*").alias("shipments"),
             F.round(F.avg("is_delayed")*100, 2).alias("delay_rate_pct"),
             F.round(F.avg("shipping_cost"), 2).alias("avg_shipping_cost")))
    route_summary.orderBy(F.desc("delay_rate_pct")).write.mode("overwrite").option("header", True).csv(str(OUT / "route_summary"))
    print(f"Spark summaries written to {OUT}")
    spark.stop()

if __name__ == "__main__":
    main()
