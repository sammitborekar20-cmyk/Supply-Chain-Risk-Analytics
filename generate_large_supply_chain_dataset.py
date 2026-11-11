
import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

    # --------------------------------------------------
    # 1. Generate shipment characteristics
    # --------------------------------------------------

    supplier = np.random.randint(1, 1001, rows)

    distance = np.random.randint(20, 3001, rows)

    traffic = np.random.choice(
        ["Low", "Medium", "High"],
        rows,
        p=[0.35, 0.40, 0.25]
    )

    weather = np.random.choice(
        ["Clear", "Rain", "Storm"],
        rows,
        p=[0.60, 0.30, 0.10]
    )

    mode = np.random.choice(
        ["Road", "Rail", "Air", "Sea"],
        rows,
        p=[0.50, 0.20, 0.15, 0.15]
    )

    inventory = np.random.randint(0, 2001, rows)

    demand = np.random.randint(50, 1501, rows)

    defective = np.random.binomial(1, 0.06, rows)

    # --------------------------------------------------
    # 2. Calculate expected delivery time
    # --------------------------------------------------

    expected_days = np.maximum(
        1,
        (distance / 500).astype(int) + 1
    )

    # --------------------------------------------------
    # 3. Create numerical risk factors
    # --------------------------------------------------

    distance_risk = (distance - 1510) / 860

    inventory_risk = (1000 - inventory) / 577

    demand_risk = (demand - 775) / 419

    expected_time_risk = (expected_days - 3.5) / 1.7

    # Inventory pressure:
    # Demand exceeding available inventory increases risk.

    inventory_pressure = np.clip(
        (demand - inventory) / 1500,
        -1,
        1
    )

    # --------------------------------------------------
    # 4. Convert categorical features into risk effects
    # --------------------------------------------------

    traffic_risk = np.select(
        [
            traffic == "High",
            traffic == "Medium"
        ],
        [
            0.90,
            0.40
        ],
        default=0.0
    )

    weather_risk = np.select(
        [
            weather == "Storm",
            weather == "Rain"
        ],
        [
            1.00,
            0.35
        ],
        default=0.0
    )

    mode_risk = np.select(
        [
            mode == "Sea",
            mode == "Rail",
            mode == "Air"
        ],
        [
            0.60,
            0.10,
            -0.40
        ],
        default=0.0
    )

    # --------------------------------------------------
    # 5. Calculate overall shipment delay probability
    # --------------------------------------------------

    risk_score = (
        -1.40
        + 0.35 * distance_risk
        + 0.35 * inventory_risk
        + 0.30 * demand_risk
        + 0.35 * expected_time_risk
        + 0.60 * inventory_pressure
        + 0.50 * defective
        + traffic_risk
        + weather_risk
        + mode_risk
    )

    # Logistic function converts risk score to probability.

    delay_probability = 1 / (1 + np.exp(-risk_score))

    # Random sampling prevents deterministic labels.

    delayed = (
        np.random.random(rows) < delay_probability
    ).astype(int)

    # --------------------------------------------------
    # 6. Generate realistic delivery delay duration
    # --------------------------------------------------

    delay_intensity = (
        0.40
        + 0.35 * np.maximum(distance_risk, 0)
        + 0.35 * np.maximum(demand_risk, 0)
        + 0.40 * np.maximum(inventory_pressure, 0)
        + 0.25 * defective
    )

    delay_intensity = np.clip(delay_intensity, 0.2, 3.0)

    delay_days = np.zeros(rows, dtype=int)

    delayed_indices = np.where(delayed == 1)[0]

    if len(delayed_indices) > 0:

        delay_days[delayed_indices] = (
            1 + np.random.poisson(
                delay_intensity[delayed_indices]
            )
        )

    actual_days = expected_days + delay_days

    delivery_delay_days = delay_days

    # --------------------------------------------------
    # 7. Generate shipment dates
    # --------------------------------------------------

    start_date = datetime(2023, 1, 1)

    date_offsets = np.random.randint(0, 1095, rows)

    dates = [
        (start_date + timedelta(days=int(offset))).strftime("%Y-%m-%d")
        for offset in date_offsets
    ]

    # --------------------------------------------------
    # 8. Generate shipping costs
    # --------------------------------------------------

    mode_cost_factor = np.select(
        [
            mode == "Road",
            mode == "Rail",
            mode == "Air",
            mode == "Sea"
        ],
        [
            1.0,
            0.75,
            2.5,
            0.55
        ],
        default=1.0
    )

    cost_per_km = np.random.uniform(1.5, 8.0, rows)

    shipping_cost = np.round(
        distance * cost_per_km * mode_cost_factor,
        2
    )

    # --------------------------------------------------
    # 9. Build final dataset
    # --------------------------------------------------

    df = pd.DataFrame({

        "shipment_id": np.arange(
            start_id,
            start_id + rows
        ),

        "supplier_id": supplier,

        "product_id": np.random.randint(1, 5001, rows),

        "warehouse_id": np.random.randint(1, 101, rows),

        "route_id": np.random.randint(1, 501, rows),

        "source": np.random.choice(
            ["Pune", "Mumbai", "Delhi", "Chennai", "Bengaluru"],
            rows
        ),

        "destination": np.random.choice(
            ["Pune", "Mumbai", "Delhi", "Chennai", "Bengaluru"],
            rows
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

        "delivery_delay_days": delivery_delay_days,

        "is_delayed": delayed,

        "defective_product": defective,

        "shipping_cost": shipping_cost,

        "month": [d[5:7] for d in dates],

        "data_source": "synthetic"
    })

    return df[COLUMNS]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total-rows",
        type=int,
        default=19_590_000
    )

    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=250_000
    )

    parser.add_argument(
        "--output",
        default="data/large_dataset_v2"
    )

    args = parser.parse_args()

    if args.total_rows <= 0 or args.rows_per_file <= 0:
        parser.error("Row counts must be positive.")

    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Do not accidentally mix old and new generated datasets.
    existing = list(output_dir.glob("shipments_part_*.csv"))

    if existing:
        raise SystemExit(
            f"Output folder already contains {len(existing)} CSV parts. "
            "Choose a new empty output folder."
        )

    print("=" * 60)
    print("SUPPLY CHAIN RISK DATASET GENERATOR V2")
    print("=" * 60)

    print(f"Target rows: {args.total_rows:,}")
    print(f"Rows per file: {args.rows_per_file:,}")
    print(f"Output: {output_dir.resolve()}")

    next_id = 1
    part_number = 1
    total_written = 0

    while total_written < args.total_rows:

        rows_to_write = min(
            args.rows_per_file,
            args.total_rows - total_written
        )

        path = output_dir / (
            f"shipments_part_{part_number:05d}.csv"
        )

        written_rows = 0

        with path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = None

            while written_rows < rows_to_write:

                batch_size = min(
                    10_000,
                    rows_to_write - written_rows
                )

                df = generate_batch(
                    next_id,
                    batch_size
                )

                df.to_csv(
                    f,
                    index=False,
                    header=(writer is None),
                    quoting=csv.QUOTE_MINIMAL
                )

                writer = True

                next_id += batch_size

                written_rows += batch_size

                total_written += batch_size

                f.flush()

                print(
                    f"\rGenerated: {total_written:,} / "
                    f"{args.total_rows:,} rows",
                    end="",
                    flush=True
                )

        print(
            f"\nCreated {path.name}: {written_rows:,} rows"
        )

        part_number += 1

    total_size = sum(
        p.stat().st_size
        for p in output_dir.glob("*.csv")
    )

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETED")
    print("=" * 60)

    print(f"Total rows: {total_written:,}")
    print(f"CSV files: {part_number - 1}")
    print(f"Total size: {total_size / 1024**3:.2f} GB")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
