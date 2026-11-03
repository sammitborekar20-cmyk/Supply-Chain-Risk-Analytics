from pathlib import Path
import numpy as np
import pandas as pd

def generate_all(root, n_shipments=20000, seed=42):
    root = Path(root)
    out = root / "data" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_suppliers = 120
    n_products = 250
    n_warehouses = 20
    n_routes = 80

    supplier_ids = [f"SUP{i:03d}" for i in range(1, n_suppliers + 1)]
    suppliers = pd.DataFrame({
        "supplier_id": supplier_ids,
        "supplier_location": rng.choice(["Pune", "Mumbai", "Nashik", "Delhi", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad"], n_suppliers),
        "supplier_rating": np.round(rng.uniform(2.0, 5.0, n_suppliers), 2),
        "base_delay_rate": np.round(rng.beta(2, 10, n_suppliers), 3),
        "defect_rate": np.round(rng.beta(1.5, 25, n_suppliers), 3),
        "reliability_score": np.round(rng.uniform(0.55, 0.99, n_suppliers), 3),
    })

    shipment_supplier = rng.choice(supplier_ids, n_shipments)
    sup_lookup = suppliers.set_index("supplier_id")
    base_delay = np.array([sup_lookup.loc[s, "base_delay_rate"] for s in shipment_supplier])
    defect = np.array([sup_lookup.loc[s, "defect_rate"] for s in shipment_supplier])
    rating = np.array([sup_lookup.loc[s, "supplier_rating"] for s in shipment_supplier])

    distance = rng.integers(20, 1800, n_shipments)
    traffic = rng.choice(["Low", "Medium", "High"], n_shipments, p=[0.45, 0.35, 0.20])
    weather = rng.choice(["Clear", "Rain", "Storm"], n_shipments, p=[0.70, 0.25, 0.05])
    mode = rng.choice(["Road", "Rail", "Air", "Sea"], n_shipments, p=[0.62, 0.20, 0.12, 0.06])
    inventory = rng.integers(0, 600, n_shipments)
    demand = rng.integers(10, 500, n_shipments)
    month = rng.integers(1, 13, n_shipments)
    expected_days = np.maximum(1, (distance / np.select([mode=="Air", mode=="Rail", mode=="Sea"], [700, 300, 180], default=250)).astype(int) + 1)
    traffic_penalty = np.select([traffic=="High", traffic=="Medium"], [0.18, 0.08], default=0)
    weather_penalty = np.select([weather=="Storm", weather=="Rain"], [0.28, 0.10], default=0)
    mode_penalty = np.select([mode=="Road", mode=="Rail", mode=="Sea"], [0.08, 0.04, 0.10], default=0.02)
    shortage_penalty = (inventory < demand * 0.25) * 0.12
    probability = np.clip(base_delay + traffic_penalty + weather_penalty + mode_penalty + shortage_penalty + (rating < 3.0)*0.10, 0.02, 0.92)
    delayed = rng.binomial(1, probability)
    extra_days = np.where(delayed == 1, rng.integers(1, 7, n_shipments), 0)
    actual_days = expected_days + extra_days
    defective = rng.binomial(1, np.clip(defect, 0, 0.5))
    source = rng.choice(["Pune", "Mumbai", "Nashik", "Delhi", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad"], n_shipments)
    destination = rng.choice(["Pune", "Mumbai", "Nashik", "Delhi", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad"], n_shipments)
    route_ids = [f"R{(i % n_routes)+1:03d}" for i in rng.integers(0, n_routes, n_shipments)]
    shipment_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n_shipments), unit="D")

    shipments = pd.DataFrame({
        "shipment_id": [f"SHP{i:07d}" for i in range(1, n_shipments+1)],
        "supplier_id": shipment_supplier,
        "product_id": [f"PRD{i:04d}" for i in rng.integers(1, n_products+1, n_shipments)],
        "warehouse_id": [f"WH{i:03d}" for i in rng.integers(1, n_warehouses+1, n_shipments)],
        "route_id": route_ids,
        "source": source,
        "destination": destination,
        "shipment_date": shipment_dates.strftime("%Y-%m-%d"),
        "transport_mode": mode,
        "distance_km": distance,
        "traffic_level": traffic,
        "weather_condition": weather,
        "inventory_level": inventory,
        "demand_units": demand,
        "expected_days": expected_days,
        "actual_days": actual_days,
        "delivery_delay_days": extra_days,
        "is_delayed": delayed,
        "defective_product": defective,
        "shipping_cost": np.round(distance * rng.uniform(1.2, 5.5, n_shipments), 2),
        "month": month,
    })

    n_inv = 250 * n_warehouses
    inventory_df = pd.DataFrame({
        "product_id": [f"PRD{i:04d}" for i in rng.integers(1, n_products+1, n_inv)],
        "warehouse_id": [f"WH{i:03d}" for i in rng.integers(1, n_warehouses+1, n_inv)],
        "stock_level": rng.integers(0, 800, n_inv),
        "reorder_level": rng.integers(50, 250, n_inv),
        "weekly_demand": rng.integers(10, 300, n_inv),
        "unit_cost": np.round(rng.uniform(5, 2500, n_inv), 2),
    })
    inventory_df["inventory_risk"] = np.where(
        inventory_df["stock_level"] < inventory_df["reorder_level"], "High",
        np.where(inventory_df["stock_level"] < inventory_df["weekly_demand"], "Medium", "Low")
    )

    suppliers.to_csv(out / "suppliers.csv", index=False)
    shipments.to_csv(out / "shipments.csv", index=False)
    inventory_df.to_csv(out / "inventory.csv", index=False)
    print(f"Generated data in {out}")
    return out
