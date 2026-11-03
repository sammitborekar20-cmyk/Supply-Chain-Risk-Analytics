from pathlib import Path
import json
import sys

from src.generate_data import generate_all
from src.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent

def main():
    n = 20000
    if len(sys.argv) > 1:
        try:
            n = max(1000, int(sys.argv[1]))
        except ValueError:
            print("Usage: python run_pipeline.py [number_of_shipments]")
            return
    print(f"[1/2] Generating demo supply-chain dataset ({n:,} shipments)...")
    generate_all(ROOT, n_shipments=n)
    print("[2/2] Cleaning data, training model, calculating risks...")
    metrics = run_pipeline(ROOT)
    print("\nPipeline completed successfully.")
    print(json.dumps(metrics, indent=2))
    print(f"\nNext: streamlit run {ROOT / 'app.py'}")

if __name__ == "__main__":
    main()
