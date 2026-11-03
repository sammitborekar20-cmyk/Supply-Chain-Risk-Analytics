"""Optional HDFS upload; Hadoop CLI must be configured and running."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
HDFS_DIR = "/supply_chain_risk/data"

def run(args):
    print("+", " ".join(args))
    subprocess.run(args, check=True)

def main():
    run(["hdfs", "dfs", "-mkdir", "-p", HDFS_DIR])
    for filename in ["suppliers.csv", "shipments.csv", "inventory.csv"]:
        path = DATA / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run python run_pipeline.py first.")
        run(["hdfs", "dfs", "-put", "-f", str(path), f"{HDFS_DIR}/{filename}"])
    run(["hdfs", "dfs", "-ls", HDFS_DIR])
    print("HDFS upload complete.")

if __name__ == "__main__":
    main()
