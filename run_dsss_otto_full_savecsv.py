import time, math, csv, os
from itertools import combinations
from pyspark.sql import SparkSession

PARQUET = "/mnt/d/DS3/data/processed/otto_transactions.parquet"
OUT_CSV = "/mnt/d/DS3/results/otto_dsss_k2_runtime.csv"
SUPPORTS = [0.001, 0.002, 0.003, 0.004, 0.005]

def run_dsss_k2(df_items, min_sup: float):
    t0 = time.time()
    m = df_items.count()
    min_count = int(math.ceil(min_sup * m))

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
                k = (a, b) if a <= b else (b, a)
                h[k] = h.get(k, 0) + 1
        for k, v in h.items():
            yield (k, v)

    counts = rdd.mapPartitions(count_partitions).reduceByKey(lambda x, y: x + y)
    freq = counts.filter(lambda kv: kv[1] >= min_count)

    freq_n = freq.count()
    top10 = freq.takeOrdered(10, key=lambda kv: -kv[1])

    dt = time.time() - t0
    return m, min_count, freq_n, dt, top10

def main():
    spark = (SparkSession.builder
             .appName("DS3_DSSS_Otto_Full_SaveCSV")
             .config("spark.sql.shuffle.partitions", "128")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(PARQUET).select("items").cache()
    df.count()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    for s in SUPPORTS:
        m, min_count, freq_n, dt, top10 = run_dsss_k2(df, s)
        print(f"[DSSS] otto sup={s}: m={m}, minCount={min_count}, freq={freq_n}, time={dt:.2f}s")
        for k, v in top10:
            print(" ", k, "->", v)
        rows.append([s, m, min_count, freq_n, round(dt, 4)])

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minSup", "m_rows", "minCount", "freq_itemsets_k<=2", "time_sec"])
        w.writerows(rows)

    print(f"[DONE] wrote: {OUT_CSV}")
    spark.stop()

if __name__ == "__main__":
    main()
