import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel

PARQUET = "/home/hi/DS3_work/data/processed/higgs_transactions.parquet"

SAMPLE_N = 20000
MIN_SUP = 0.005
PARTS = 64

def main():
    spark = (SparkSession.builder
             .appName("DS3_exact_pairs_baseline")
             .config("spark.sql.shuffle.partitions", str(PARTS))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df0 = spark.read.parquet(PARQUET).select("items")

    # Deterministic "first N rows" using zipWithIndex
    rdd = df0.rdd.zipWithIndex().filter(lambda x: x[1] < SAMPLE_N).map(lambda x: x[0])
    df = spark.createDataFrame(rdd, df0.schema).repartition(PARTS).persist(StorageLevel.MEMORY_AND_DISK)
    m = df.count()
    minCount = int((MIN_SUP * m) + 0.999999999)  # ceil
    print(f"[INFO] sample m={m}, minSup={MIN_SUP}, minCount={minCount}")

    # 1-item exact counts
    t0 = time.time()
    one = (df.select(F.explode("items").alias("i"))
             .groupBy("i").count()
             .filter(F.col("count") >= minCount))
    one_count = one.count()
    one_top = one.orderBy(F.desc("count")).limit(10).collect()
    t1 = time.time()
    print(f"[EXACT-1] frequent_1_itemsets={one_count}, time_sec={t1-t0:.2f}")
    print("[EXACT-1] top10 (item -> count):")
    for r in one_top:
        print(f"   ({r['i']},) -> {r['count']}")

    # 2-item exact counts (pairs per transaction)
    t0 = time.time()
    pair_df = (df
        .select(F.sort_array("items").alias("items"))
        .withColumn(
            "pairs",
            F.expr("""
            transform(
              sequence(0, size(items)-2),
              x -> transform(sequence(x+1, size(items)-1), y -> array(items[x], items[y]))
            )
            """)
        )
        .select(F.explode(F.flatten("pairs")).alias("pair"))
        .groupBy("pair").count()
        .filter(F.col("count") >= minCount)
    )
    pair_count = pair_df.count()
    pair_top = pair_df.orderBy(F.desc("count")).limit(10).collect()
    t1 = time.time()
    print(f"[EXACT-2] frequent_2_itemsets={pair_count}, time_sec={t1-t0:.2f}")
    print("[EXACT-2] top10 (pair -> count):")
    for r in pair_top:
        a, b = r["pair"][0], r["pair"][1]
        print(f"   ({a}, {b}) -> {r['count']}")

    print(f"[EXACT] total_k<=2={one_count + pair_count}")

    spark.stop()

if __name__ == "__main__":
    main()
