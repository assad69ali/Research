import time, math
from itertools import combinations
from pyspark.sql import SparkSession

PARQUET = "/mnt/d/DS3/data/processed/higgs_transactions.parquet"
MIN_SUP = 0.005   # paper uses 0.5%
KMAX = 2

CORES_LIST = [2, 4, 8, 16]
OUT_CSV = "/mnt/d/DS3/results/higgs_scaling_localcores.csv"

def dsss_k2_count(df_items, min_sup: float):
    m = df_items.count()
    min_count = int(math.ceil(min_sup * m))
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
                .reduceByKey(lambda x, y: x + y)
                .filter(lambda kv: kv[1] >= min_count))
    # force execution
    nfreq = counts.count()
    return m, min_count, nfreq

def main():
    with open(OUT_CSV, "w") as f:
        f.write("cores,m_rows,minSup,minCount,freq_itemsets_k<=2,time_sec\n")

    for c in CORES_LIST:
        spark = (SparkSession.builder
                 .appName(f"higgs_scaling_{c}cores")
                 .master(f"local[{c}]")
                 .config("spark.sql.shuffle.partitions", str(max(8, c*2)))
                 .getOrCreate())
        spark.sparkContext.setLogLevel("WARN")

        df = spark.read.parquet(PARQUET).select("items")

        t0 = time.time()
        m, min_count, nfreq = dsss_k2_count(df, MIN_SUP)
        t1 = time.time()
        dt = t1 - t0

        print(f"[SCALING] cores={c} m={m} minSup={MIN_SUP} minCount={min_count} freq={nfreq} time={dt:.3f}s")

        with open(OUT_CSV, "a") as f:
            f.write(f"{c},{m},{MIN_SUP},{min_count},{nfreq},{dt:.6f}\n")

        spark.stop()

    print("[DONE] wrote:", OUT_CSV)

if __name__ == "__main__":
    main()
