from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

CATEGORICAL = ["transport_mode", "traffic_level", "weather_condition"]
NUMERIC = ["distance_km", "inventory_level", "demand_units", "expected_days", "shipping_cost", "month",
           "supplier_rating", "supplier_delay_rate", "supplier_defect_rate", "supplier_reliability"]

def run_pipeline(root):
    root = Path(root)
    data_dir = root / "data" / "generated"
    out_dir = root / "outputs"
    model_dir = root / "models"
    out_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    shipments = pd.read_csv(data_dir / "shipments.csv")
    suppliers = pd.read_csv(data_dir / "suppliers.csv")
    inventory = pd.read_csv(data_dir / "inventory.csv")

    # Data cleaning and supplier-level historical risk features.
    for col in ["distance_km", "inventory_level", "demand_units", "shipping_cost"]:
        shipments[col] = pd.to_numeric(shipments[col], errors="coerce")
    supplier_stats = shipments.groupby("supplier_id").agg(
        supplier_delay_rate=("is_delayed", "mean"),
        supplier_avg_delay=("delivery_delay_days", "mean"),
    ).reset_index()
    enriched = shipments.merge(suppliers, on="supplier_id", how="left")
    enriched = enriched.merge(supplier_stats, on="supplier_id", how="left")
    enriched["supplier_delay_rate"] = enriched["supplier_delay_rate"].fillna(0)
    enriched["supplier_defect_rate"] = enriched["defect_rate"].fillna(0)
    enriched["supplier_reliability"] = enriched["reliability_score"].fillna(0.8)

    # Avoid using the target or post-delivery values as features.
    X = enriched[CATEGORICAL + NUMERIC].copy()
    y = enriched["is_delayed"].astype(int)
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocess", pre),
        ("classifier", RandomForestClassifier(n_estimators=140, max_depth=12, min_samples_leaf=3,
                                              class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "rows_processed": int(len(enriched)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4) if y.nunique() == 2 else None,
        "note": "Metrics are from synthetic demo data and must not be interpreted as real-world performance."
    }
    joblib.dump(model, model_dir / "shipment_delay_model.joblib")

    # Predict all rows for dashboard use.
    enriched["delay_probability"] = model.predict_proba(X)[:, 1]
    enriched["supplier_risk_component"] = (1 - enriched["reliability_score"].clip(0, 1)) * 100
    enriched["quality_risk_component"] = enriched["defect_rate"].clip(0, 1) * 100
    enriched["inventory_risk_component"] = np.where(
        enriched["inventory_level"] < enriched["demand_units"] * 0.25, 100,
        np.where(enriched["inventory_level"] < enriched["demand_units"], 60, 20)
    )
    enriched["external_risk_component"] = (
        enriched["weather_condition"].map({"Clear": 10, "Rain": 50, "Storm": 95}).fillna(20) * 0.6
        + enriched["traffic_level"].map({"Low": 10, "Medium": 50, "High": 90}).fillna(20) * 0.4
    )
    # Weighted score: prediction (delivery risk), quality, supplier, inventory, external.
    enriched["risk_score"] = (
        enriched["delay_probability"] * 100 * 0.30
        + enriched["quality_risk_component"] * 0.25
        + enriched["supplier_risk_component"] * 0.20
        + enriched["inventory_risk_component"] * 0.15
        + enriched["external_risk_component"] * 0.10
    ).clip(0, 100).round(1)
    enriched["risk_level"] = pd.cut(enriched["risk_score"], bins=[-0.1, 30, 60, 100],
                                    labels=["Low", "Medium", "High"]).astype(str)
    enriched["predicted_delay"] = (enriched["delay_probability"] >= 0.5).astype(int)

    shipment_cols = ["shipment_id", "supplier_id", "product_id", "warehouse_id", "route_id", "source", "destination",
                     "transport_mode", "distance_km", "traffic_level", "weather_condition", "inventory_level",
                     "demand_units", "is_delayed", "delivery_delay_days", "delay_probability", "predicted_delay",
                     "risk_score", "risk_level", "shipping_cost"]
    enriched[shipment_cols].to_csv(out_dir / "shipment_risk_predictions.csv", index=False)

    supplier_summary = enriched.groupby("supplier_id").agg(
        shipments=("shipment_id", "count"),
        delay_rate=("is_delayed", "mean"),
        average_delay_days=("delivery_delay_days", "mean"),
        average_risk_score=("risk_score", "mean"),
        defect_rate=("defective_product", "mean"),
        supplier_rating=("supplier_rating", "mean"),
        reliability_score=("reliability_score", "mean"),
    ).reset_index()
    supplier_summary["delay_rate"] = (supplier_summary["delay_rate"] * 100).round(1)
    supplier_summary["defect_rate"] = (supplier_summary["defect_rate"] * 100).round(1)
    supplier_summary["average_delay_days"] = supplier_summary["average_delay_days"].round(2)
    supplier_summary["average_risk_score"] = supplier_summary["average_risk_score"].round(1)
    supplier_summary["risk_level"] = pd.cut(supplier_summary["average_risk_score"], [-0.1, 30, 60, 100],
                                             labels=["Low", "Medium", "High"]).astype(str)
    supplier_summary.sort_values("average_risk_score", ascending=False).to_csv(out_dir / "supplier_risk_summary.csv", index=False)

    route_summary = enriched.groupby("route_id").agg(
        shipments=("shipment_id", "count"),
        delay_rate=("is_delayed", "mean"),
        average_delay_days=("delivery_delay_days", "mean"),
        average_risk_score=("risk_score", "mean"),
        average_distance_km=("distance_km", "mean"),
        shipping_cost=("shipping_cost", "mean"),
    ).reset_index()
    route_summary["delay_rate"] = (route_summary["delay_rate"] * 100).round(1)
    route_summary["average_delay_days"] = route_summary["average_delay_days"].round(2)
    route_summary["average_risk_score"] = route_summary["average_risk_score"].round(1)
    route_summary["average_distance_km"] = route_summary["average_distance_km"].round(1)
    route_summary.sort_values("average_risk_score", ascending=False).to_csv(out_dir / "route_risk_summary.csv", index=False)

    inventory["days_of_cover"] = (inventory["stock_level"] / inventory["weekly_demand"].clip(lower=1) * 7).round(1)
    inventory["inventory_risk"] = np.where(
        inventory["stock_level"] < inventory["reorder_level"], "High",
        np.where(inventory["stock_level"] < inventory["weekly_demand"], "Medium", "Low")
    )
    inventory.sort_values(["inventory_risk", "days_of_cover"]).to_csv(out_dir / "inventory_risk_summary.csv", index=False)
    with open(out_dir / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
