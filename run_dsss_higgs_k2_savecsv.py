import time, math, csv
from itertools import combinations

from pyspark.sql import SparkSession

PARQUET = "/mnt/d/DS3/data/processed/higgs_transactions.parquet"
OUTCSV  = "/mnt/d/DS3/results/higgs_dsss_k2_summary.csv"
SUPPORTS = [0.001, 0.002, 0.003, 0.004, 0.005]

def run_one(df_items, min_sup: float, tag: str):
    t0 = time.time()
    m = df_items.count()
    min_count = int(math.ceil(min_sup * m))
    print(f"[INFO] {tag}: m={m}, minSup={min_sup}, minCount={min_count}")

    rdd = df_items.rdd.map(lambda r: r["items"])

    def count_partitions(it):
        h = {}
        for items in it:
            if not items:
                continue
            for a in items:
                k = (a,)
                h[k] = h.get(k, 0) + 1
            for a, b in combinations(items, 2):
                if a <= b:
                    k = (a, b)
                else:
                    k = (b, a)
                h[k] = h.get(k, 0) + 1
        for k, v in h.items():
            yield (k, v)

    counts = rdd.mapPartitions(count_partitions).reduceByKey(lambda x, y: x + y)
    frequent = counts.filter(lambda kv: kv[1] >= min_count)

    nfreq = frequent.count()
    dt = time.time() - t0
    print(f"[DSSS] {tag}: frequent_itemsets={nfreq}, time_sec={dt:.2f}")
    return (min_sup, m, min_count, nfreq, dt)

def main():
    spark = (SparkSession.builder
             .appName("DS3_DSSS_HIGGS_k2_savecsv")
             .config("spark.sql.shuffle.partitions", "128")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(PARQUET).select("items").cache()
    df.count()

    rows = []
    for s in SUPPORTS:
        rows.append(run_one(df, s, tag=f"higgs_subset_sup_{s}"))

    spark.stop()

    with open(OUTCSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minSup","m_rows","minCount","freq_itemsets_k<=2","time_sec"])
        for r in rows:
            w.writerow(r)

    print("[DONE] wrote:", OUTCSV)

if __name__ == "__main__":
    main()
