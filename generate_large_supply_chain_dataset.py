# generate_large_supply_chain_dataset.py

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

COLUMNS = [
    "shipment_id", "supplier_id", "product_id", "warehouse_id",
    "route_id", "source", "destination", "shipment_date",
    "transport_mode", "distance_km", "traffic_level",
    "weather_condition", "inventory_level", "demand_units",
    "expected_days", "actual_days", "delivery_delay_days",
    "is_delayed", "defective_product", "shipping_cost", "month",
    "data_source"
]

def generate_batch(start_id, rows):
    supplier = np.random.randint(1, 1001, rows)
    distance = np.random.randint(20, 3001, rows)
    traffic = np.random.choice(
        ["Low", "Medium", "High"], rows, p=[0.35, 0.40, 0.25]
    )
    weather = np.random.choice(
        ["Clear", "Rain", "Storm"], rows, p=[0.60, 0.30, 0.10]
    )
    mode = np.random.choice(
        ["Road", "Rail", "Air", "Sea"], rows, p=[0.50, 0.20, 0.15, 0.15]
    )

    expected_days = np.maximum(1, (distance / 500).astype(int) + 1)
    traffic_days = np.select(
        [traffic == "High", traffic == "Medium"],
        [2, 1], default=0
    )
    weather_days = np.select(
        [weather == "Storm", weather == "Rain"],
        [3, 1], default=0
    )
    mode_adjustment = np.select(
        [mode == "Air", mode == "Rail", mode == "Sea"],
        [-1, 0, 3], default=0
    )

    noise = np.random.randint(-1, 3, rows)
    actual_days = np.maximum(
        1, expected_days + traffic_days + weather_days + mode_adjustment + noise
    )
    delay = np.maximum(0, actual_days - expected_days)
    delayed = (delay > 0).astype(int)

    dates = [
        (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 1094)))
        .strftime("%Y-%m-%d")
        for _ in range(rows)
    ]

    inventory = np.random.randint(0, 2001, rows)
    demand = np.random.randint(50, 1501, rows)
    defective = np.random.binomial(1, 0.06, rows)

    df = pd.DataFrame({
        "shipment_id": np.arange(start_id, start_id + rows),
        "supplier_id": supplier,
        "product_id": np.random.randint(1, 5001, rows),
        "warehouse_id": np.random.randint(1, 101, rows),
        "route_id": np.random.randint(1, 501, rows),
        "source": np.random.choice(
            ["Pune", "Mumbai", "Delhi", "Chennai", "Bengaluru"], rows
        ),
        "destination": np.random.choice(
            ["Pune", "Mumbai", "Delhi", "Chennai", "Bengaluru"], rows
        ),
        "shipment_date": dates,
        "transport_mode": mode,
        "distance_km": distance,
        "traffic_level": traffic,
        "weather_condition": weather,
        "inventory_level": inventory,
        "demand_units": demand,
        "expected_days": expected_days,
        "actual_days": actual_days,
        "delivery_delay_days": delay,
        "is_delayed": delayed,
        "defective_product": defective,
        "shipping_cost": np.round(
            distance * np.random.uniform(1.5, 8.0, rows), 2
        ),
        "month": [d[5:7] for d in dates],
        "data_source": "synthetic"
    })
    return df[COLUMNS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gb", type=float, default=2.0)
    parser.add_argument("--rows-per-file", type=int, default=250000)
    parser.add_argument("--output", default="data/large_dataset")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_gb * 1024**3)
    total_bytes = sum(p.stat().st_size for p in output_dir.glob("*.csv"))
    next_id = 1
    existing = sorted(output_dir.glob("shipments_part_*.csv"))

    if existing:
        # Avoid overwriting old files and continue IDs after existing files.
        next_part = len(existing) + 1
        next_id = (next_part - 1) * args.rows_per_file + 1
    else:
        next_part = 1

    print(f"Target size: {args.target_gb} GB")
    print(f"Existing CSV size: {total_bytes / 1024**3:.2f} GB")

    while total_bytes < target_bytes:
        path = output_dir / f"shipments_part_{next_part:05d}.csv"
        written_rows = 0

        # Check file size after each batch, not after every row.
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = None
            while total_bytes < target_bytes and written_rows < args.rows_per_file:
                batch_size = min(10000, args.rows_per_file - written_rows)
                df = generate_batch(next_id, batch_size)
                df.to_csv(
                    f,
                    index=False,
                    header=(writer is None),
                    quoting=csv.QUOTE_MINIMAL
                )
                writer = True
                next_id += batch_size
                written_rows += batch_size
                f.flush()
                total_bytes = sum(
                    p.stat().st_size for p in output_dir.glob("*.csv")
                )
                print(
                    f"\rSize: {total_bytes / 1024**3:.2f} GB",
                    end="",
                    flush=True
                )

        print(f"\nCreated {path.name}: {written_rows:,} rows")
        next_part += 1

    print(f"\nFinished. CSV data is in: {output_dir.resolve()}")
    print(f"Total CSV size: {total_bytes / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()