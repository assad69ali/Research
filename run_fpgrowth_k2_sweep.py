import os, time, math, csv
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

DS3_ROOT = "/mnt/d/DS3"

def get_paths(dataset: str):
    if dataset == "otto":
        return f"{DS3_ROOT}/data/processed/otto_transactions.parquet", f"{DS3_ROOT}/results/otto_fpgrowth_k2_runtime.csv"
    elif dataset == "higgs":
        return f"{DS3_ROOT}/data/processed/higgs_transactions.parquet", f"{DS3_ROOT}/results/higgs_fpgrowth_k2_runtime.csv"
    else:
        raise ValueError("DATASET must be 'otto' or 'higgs'")

def run_one(spark, parquet_path: str, minSup: float, max_k: int = 2, num_parts=None):
    df = spark.read.parquet(parquet_path).select("items")
    if num_parts is not None and int(num_parts) > 0:
        df = df.repartition(int(num_parts))

    m = df.count()
    minCount = int(math.ceil(m * minSup))

    t0 = time.time()
    fp = FPGrowth(itemsCol="items", minSupport=minSup)  # DO NOT set numPartitions (causes 0-bug)
    model = fp.fit(df)

    fi2 = model.freqItemsets.where(F.size("items") <= int(max_k))
    freq_cnt = fi2.count()
    t1 = time.time()

    return {
        "minSup": minSup,
        "m_rows": m,
        "minCount": minCount,
        "freq_itemsets_k<=2": freq_cnt,
        "time_sec": round(t1 - t0, 6),
    }

def main():
    dataset = os.getenv("DATASET", "otto").strip().lower()
    parquet_path, out_csv = get_paths(dataset)

    master = os.getenv("SPARK_MASTER", "local[*]")
    num_parts_env = os.getenv("NUM_PARTS", "").strip()
    num_parts = int(num_parts_env) if num_parts_env.isdigit() else None

    spark = (SparkSession.builder
        .appName(f"DS3_FPGrowth_sweep_{dataset}")
        .master(master)
        .config("spark.local.dir", f"{DS3_ROOT}/tmp")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    sups = [0.001, 0.002, 0.003, 0.004, 0.005]
    max_k = int(os.getenv("MAX_K", "2"))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minSup","m_rows","minCount","freq_itemsets_k<=2","time_sec"])
        for s in sups:
            print(f"[INFO] FPGrowth {dataset}: minSup={s}, max_k={max_k}, NUM_PARTS={num_parts}")
            res = run_one(spark, parquet_path, s, max_k=max_k, num_parts=num_parts)
            print("[FPG]", res)
            w.writerow([res["minSup"], res["m_rows"], res["minCount"], res["freq_itemsets_k<=2"], res["time_sec"]])

    spark.stop()
    print(f"[DONE] wrote {out_csv}")

if __name__ == "__main__":
    main()
