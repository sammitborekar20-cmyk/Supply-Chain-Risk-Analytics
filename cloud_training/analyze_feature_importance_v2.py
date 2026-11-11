
from pathlib import Path
import csv

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "full_dataset_spark_ml_v2_dataset_v2"
    / "supply_chain_rf_full"
)

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "feature_importance"
    / "feature_importance_v2.csv"
)

spark = (
    SparkSession.builder
    .appName("SupplyChainV2FeatureImportance")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print("\nLoading V2 trained model...")

    model = PipelineModel.load(str(MODEL_PATH))

    # The final pipeline stage is the Random Forest model.
    rf_model = model.stages[-1]

    # The preceding stage is the VectorAssembler.
    assembler = model.stages[-2]

    feature_names = assembler.getInputCols()
    importance_values = rf_model.featureImportances.toArray()

    if len(feature_names) != len(importance_values):
        raise ValueError(
            "Feature names and importance values do not match."
        )

    results = sorted(
        zip(feature_names, importance_values),
        key=lambda item: item[1],
        reverse=True
    )

    print("\n========== V2 FEATURE IMPORTANCE ==========")

    for name, importance in results:
        print(f"{name:30s}: {importance * 100:.4f}%")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "importance", "importance_percentage"])

        for name, importance in results:
            writer.writerow([
                name,
                importance,
                round(importance * 100, 4)
            ])

    print("\nFeature importance saved to:")
    print(OUTPUT_PATH)

finally:
    spark.stop()
