import time, math
from itertools import combinations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PARQUET = "/mnt/d/DS3/data/processed/higgs_transactions.parquet"
SUPPORTS = [0.001, 0.002, 0.003, 0.004, 0.005]
KMAX = 2

def run_dsss_k2(df_items, min_sup: float, tag: str):
    t0 = time.time()

    m = df_items.count()
    min_count = int(math.ceil(min_sup * m))
    print(f"[INFO] {tag}: m={m}, minSup={min_sup}, minCount={min_count}, kmax={KMAX}")

    rdd = df_items.rdd.map(lambda r: r["items"])

    def count_partitions(it):
        h = {}
        for items in it:
            if not items:
                continue

            # 1-itemsets
            for a in items:
                k = (a,)
                h[k] = h.get(k, 0) + 1

            # 2-itemsets
            for a, b in combinations(items, 2):
                if a <= b:
                    k = (a, b)
                else:
                    k = (b, a)
                h[k] = h.get(k, 0) + 1

        for k, v in h.items():
            yield (k, v)

    counts = (rdd.mapPartitions(count_partitions)
                .reduceByKey(lambda x, y: x + y))

    frequent = counts.filter(lambda kv: kv[1] >= min_count)

    nfreq = frequent.count()
    top10 = frequent.takeOrdered(10, key=lambda kv: -kv[1])

    dt = time.time() - t0
    print(f"[DSSS] {tag}: frequent_itemsets={nfreq}, time_sec={dt:.2f}")
    print(f"[DSSS] {tag}: top10 (itemset -> count):")
    for k, v in top10:
        print(f"   {k} -> {v}")

    return {"m": m, "min_sup": min_sup, "min_count": min_count, "nfreq": nfreq, "time_sec": dt}

def main():
    spark = (SparkSession.builder
             .appName("DS3_DSSS_HIGGS_k2")
             .config("spark.sql.shuffle.partitions", "128")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(PARQUET).select("items").cache()
    df.count()  # materialize cache

    for s in SUPPORTS:
        run_dsss_k2(df, s, tag=f"higgs_subset_sup_{s}")

    spark.stop()

if __name__ == "__main__":
    main()
