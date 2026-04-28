from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW_TRAIN = "/mnt/d/DS3/data/raw/train.csv"

# Write outputs to Linux filesystem (WSL) to avoid /mnt/d permission issues
OUT_PARQUET = "/home/hi/DS3_work/otto_transactions.parquet"
STATS_OUT   = "/home/hi/DS3_work/dataset_stats_otto.txt"

BINS = 10
REL_ERROR = 0.001
DROP = {"id", "target"}

def make_bin_col(colname, cutpoints):
    c = F.col(colname)
    expr = F.lit(0)
    for i, cp in enumerate(cutpoints):
        expr = F.when(c > F.lit(cp), F.lit(i + 1)).otherwise(expr)
    return expr

def main():
    spark = (SparkSession.builder
             .appName("DS3_prepare_transactions_otto")
             .config("spark.local.dir", "/home/hi/DS3_work/tmp")
             .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.option("header", True).option("inferSchema", True).csv(RAW_TRAIN)

    features = []
    for c, dtype in df.dtypes:
        if c in DROP:
            continue
        if dtype in ("double","float","int","bigint","smallint","tinyint"):
            features.append(c)

    for c in features:
        df = df.withColumn(c, F.col(c).cast("double"))

    probs = [i / BINS for i in range(1, BINS)]
    item_cols = []
    for c in features:
        cps = df.approxQuantile(c, probs, REL_ERROR)
        b = make_bin_col(c, cps).cast("int")
        item_cols.append(F.concat(F.lit(f"{c}_q"), b.cast("string")))

    out = df.select(F.array(*item_cols).alias("items"))
    out.write.mode("overwrite").parquet(OUT_PARQUET)

    m = out.count()
    n = out.select(F.explode("items").alias("item")).distinct().count()
    lens = out.select(F.size("items").alias("len"))
    avg_len = float(lens.agg(F.avg("len")).first()[0])
    max_len = int(lens.agg(F.max("len")).first()[0])

    with open(STATS_OUT, "w") as f:
        f.write(f"dataset=Otto\nm={m}\nn={n}\navg_len={avg_len}\nmax_len={max_len}\n")
        f.write(f"encoding=quantile_bins={BINS}, relError={REL_ERROR}\n")

    print("[DONE] parquet:", OUT_PARQUET)
    print("[DONE] stats:", STATS_OUT)
    spark.stop()

if __name__ == "__main__":
    main()
