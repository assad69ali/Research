import time, math
from itertools import combinations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PARQUET = "/home/hi/DS3_work/otto_transactions.parquet"
SUPPORTS = [0.001, 0.002, 0.003, 0.004, 0.005]
KMAX = 2

def run_dsss_k2(df_items, min_sup: float, tag: str):
    """
    DSSS-style: mapPartitions local counting + reduceByKey global merge.
    Here we cap at k<=2 for feasibility (matches your validated baseline).
    """
    t0 = time.time()

    m = df_items.count()
    min_count = int(math.ceil(min_sup * m))
    print(f"[INFO] {tag}: m={m}, minSup={min_sup}, minCount={min_count}, kmax={KMAX}")

    rdd = df_items.rdd.map(lambda r: r["items"])

    def count_partitions(it):
        # local hash table (dict)
        h = {}
        for items in it:
            # safety: ensure list
            if not items:
                continue

            # 1-itemsets
            for a in items:
                k = (a,)
                h[k] = h.get(k, 0) + 1

            # 2-itemsets
            # IMPORTANT: items are already strings like feat_.._q..
            # Ensure deterministic ordering
            # (combinations uses input order, but your items should already be stable)
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
    freq_count = frequent.count()

    top10 = frequent.takeOrdered(10, key=lambda kv: -kv[1])

    t1 = time.time()
    print(f"[DSSS] {tag}: frequent_itemsets={freq_count}, time_sec={t1 - t0:.2f}")
    print(f"[DSSS] {tag}: top10 (itemset -> count):")
    for k, v in top10:
        print(f"   {k} -> {v}")

    return {"tag": tag, "m": m, "min_sup": min_sup, "min_count": min_count,
            "frequent_itemsets": freq_count, "time_sec": (t1 - t0)}

def main():
    spark = (SparkSession.builder
             .appName("DS3_DSSS_Otto_Full_k2")
             .config("spark.sql.shuffle.partitions", "128")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Read parquet
    df0 = spark.read.parquet(PARQUET).select("items")

    # good practice: repartition + cache once
    df = df0.repartition(128).cache()
    _ = df.count()

    results = []
    for s in SUPPORTS:
        results.append(run_dsss_k2(df, s, tag=f"otto_full_sup_{s}"))

    # write a tiny results table
    out = spark.createDataFrame(results)
    out.coalesce(1).write.mode("overwrite").option("header", True).csv("/home/hi/DS3_work/results/otto_dsss_k2_runs_csv")

    spark.stop()

if __name__ == "__main__":
    main()
