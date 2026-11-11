
import csv
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "full_dataset_spark_ml_v2"
    / "supply_chain_rf_full"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "feature_importance"

CSV_PATH = OUTPUT_DIR / "feature_importance.csv"
JSON_PATH = OUTPUT_DIR / "feature_importance.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SupplyChainFeatureImportance")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:

    # --------------------------------------------------
    # 3. CHECK MODEL PATH
    # --------------------------------------------------

    print("\nLoading existing trained model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )

    # Load the previously trained pipeline
    model = PipelineModel.load(str(MODEL_PATH))

    print("Model loaded successfully!")


    # --------------------------------------------------
    # 4. EXTRACT RANDOM FOREST MODEL
    # --------------------------------------------------

    rf_model = model.stages[-1]

    if not hasattr(rf_model, "featureImportances"):
        raise TypeError(
            "The final pipeline stage does not contain "
            "Random Forest feature importances."
        )

    feature_importances = (
        rf_model.featureImportances.toArray()
    )


    # --------------------------------------------------
    # 5. IDENTIFY VECTOR ASSEMBLER
    # --------------------------------------------------

    assembler = next(
        (
            stage
            for stage in model.stages
            if isinstance(stage, VectorAssembler)
        ),
        None
    )

    if assembler is None:
        raise ValueError(
            "VectorAssembler stage not found in the trained model."
        )

    feature_names = assembler.getInputCols()


    # --------------------------------------------------
    # 6. MATCH FEATURES WITH IMPORTANCE SCORES
    # --------------------------------------------------

    if len(feature_names) != len(feature_importances):
        raise ValueError(
            "Feature names and importance scores do not match. "
            f"Features: {len(feature_names)}, "
            f"Importance scores: {len(feature_importances)}"
        )

    results = []

    for name, importance in zip(
        feature_names,
        feature_importances
    ):

        results.append({
            "feature": name,
            "importance": float(importance),
            "importance_percentage": float(importance * 100)
        })


    # --------------------------------------------------
    # 7. SORT BY IMPORTANCE
    # --------------------------------------------------

    results.sort(
        key=lambda item: item["importance"],
        reverse=True
    )


    # --------------------------------------------------
    # 8. SAVE RESULTS AS CSV
    # --------------------------------------------------

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "feature",
                "importance",
                "importance_percentage"
            ]
        )

        writer.writeheader()
        writer.writerows(results)


    # --------------------------------------------------
    # 9. SAVE RESULTS AS JSON
    # --------------------------------------------------

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8"
    ) as json_file:

        json.dump(
            results,
            json_file,
            indent=4
        )


    # --------------------------------------------------
    # 10. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("SUPPLY CHAIN RISK ANALYTICS")
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 55)

    print(
        f"\n{'Feature':<30}"
        f"{'Importance (%)':>20}"
    )

    print("-" * 50)

    for item in results:

        print(
            f"{item['feature']:<30}"
            f"{item['importance_percentage']:>19.2f}%"
        )

    print("-" * 50)

    print("\nFeature importance analysis completed!")

    print(f"\nCSV saved at:\n{CSV_PATH}")
    print(f"\nJSON saved at:\n{JSON_PATH}")


finally:

    # --------------------------------------------------
    # 11. STOP SPARK
    # --------------------------------------------------

    spark.stop()

    print("\nSpark session stopped.")
