import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DS3_ROOT = "/mnt/d/DS3"
IN_CSV = f"{DS3_ROOT}/data/raw/higgs_subset_header.csv"
OUT_PARQUET = f"{DS3_ROOT}/data/processed/higgs_transactions.parquet"

BINS = 10
REL_ERROR = 0.001
DROP_COLS = {"label"}

def main():
    spark = (SparkSession.builder
             .appName("DS3_prepare_higgs_subset")
             .config("spark.local.dir", f"{DS3_ROOT}/tmp")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = (spark.read.option("header", True).option("inferSchema", True).csv(IN_CSV))

    # numeric feature columns
    feats = [c for c, t in df.dtypes if c not in DROP_COLS and t in ("double","float","int","bigint","smallint","tinyint")]
    for c in feats:
        df = df.withColumn(c, F.col(c).cast("double"))

    # compute quantile cutpoints per feature
    probs = [i / BINS for i in range(1, BINS)]
    cut_map = {}
    for c in feats:
        qs = df.approxQuantile(c, probs, REL_ERROR)
        # dedupe + monotonic
        uniq = []
        last = None
        for q in qs:
            if last is None or q > last:
                uniq.append(q); last = q
        cut_map[c] = uniq

    # bin column builder (0..BINS-1)
    def bin_expr(colname, cuts):
        c = F.col(colname)
        expr = F.lit(0)
        for i, cp in enumerate(cuts):
            expr = F.when(c > F.lit(cp), F.lit(i+1)).otherwise(expr)
        return expr

    # build items array: f{i}_q{bin}
    item_cols = []
    for c in feats:
        cuts = cut_map[c]
        b = bin_expr(c, cuts)
        item_cols.append(F.concat(F.lit(f"{c}_q"), b.cast("string")))

    out = df.select(F.array(*item_cols).alias("items"))
    out.write.mode("overwrite").parquet(OUT_PARQUET)

    print(f"[DONE] parquet: {OUT_PARQUET}")
    spark.stop()

if __name__ == "__main__":
    main()
