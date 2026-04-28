import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/mnt/d/DS3"
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("[SAVED]", path)

def plot_dsss_vs_fpg_higgs():
    dsss_path = os.path.join(ROOT, "results", "higgs_dsss_k2_summary.csv")
    fpg_path  = os.path.join(ROOT, "results", "higgs_fpgrowth_k2_runtime.csv")

    if not (os.path.exists(dsss_path) and os.path.exists(fpg_path)):
        print("[SKIP] Missing:", dsss_path, "or", fpg_path)
        return

    dsss = pd.read_csv(dsss_path)
    fpg  = pd.read_csv(fpg_path)

    # Merge on minSup (and minCount if present)
    key_cols = ["minSup"]
    if "minCount" in dsss.columns and "minCount" in fpg.columns:
        key_cols.append("minCount")

    m = pd.merge(dsss, fpg, on=key_cols, suffixes=("_dsss", "_fpg"))
    # Expect columns: time_sec_dsss, time_sec_fpg
    if "time_sec_dsss" not in m.columns or "time_sec_fpg" not in m.columns:
        print("[SKIP] Unexpected columns in merge. Columns:", m.columns.tolist())
        return

    plt.figure()
    plt.plot(m["minSup"], m["time_sec_dsss"], marker="o", label="DSSS (k<=2)")
    plt.plot(m["minSup"], m["time_sec_fpg"],  marker="o", label="FP-Growth (k<=2)")
    plt.xlabel("minSup")
    plt.ylabel("time (sec)")
    plt.title("HIGGS: Runtime vs Support (k<=2)")
    plt.legend()
    save_plot(os.path.join(PLOTS, "higgs_runtime_dsss_vs_fpgrowth.png"))

    # Speedup plot (FP / DSSS)
    plt.figure()
    speedup = m["time_sec_fpg"] / m["time_sec_dsss"]
    plt.plot(m["minSup"], speedup, marker="o")
    plt.xlabel("minSup")
    plt.ylabel("speedup = FP / DSSS")
    plt.title("HIGGS: Speedup of DSSS over FP-Growth (k<=2)")
    save_plot(os.path.join(PLOTS, "higgs_speedup_fpg_over_dsss.png"))

def plot_otto_dsss_runtime():
    otto_path = os.path.join(ROOT, "results", "otto_dsss_k2_runtime.csv")
    if not os.path.exists(otto_path):
        print("[SKIP] Missing:", otto_path)
        return

    df = pd.read_csv(otto_path)
    if "minSup" not in df.columns or "time_sec" not in df.columns:
        print("[SKIP] Unexpected columns:", df.columns.tolist())
        return

    plt.figure()
    plt.plot(df["minSup"], df["time_sec"], marker="o")
    plt.xlabel("minSup")
    plt.ylabel("time (sec)")
    plt.title("OTTO: DSSS Runtime vs Support (k<=2)")
    save_plot(os.path.join(PLOTS, "otto_dsss_runtime_vs_support.png"))

def plot_higgs_scaling():
    s_path = os.path.join(ROOT, "results", "higgs_scaling_localcores.csv")
    if not os.path.exists(s_path):
        print("[SKIP] Missing:", s_path)
        return

    df = pd.read_csv(s_path)
    if "cores" not in df.columns or "time_sec" not in df.columns:
        print("[SKIP] Unexpected columns:", df.columns.tolist())
        return

    plt.figure()
    plt.plot(df["cores"], df["time_sec"], marker="o")
    plt.xlabel("local cores")
    plt.ylabel("time (sec)")
    plt.title("HIGGS: DSSS Scaling (local cores)")
    save_plot(os.path.join(PLOTS, "higgs_dsss_scaling_localcores.png"))

def main():
    plot_dsss_vs_fpg_higgs()
    plot_otto_dsss_runtime()
    plot_higgs_scaling()

    print("\n[DONE] Plots saved in:", PLOTS)
    print("List them with: ls -lh", PLOTS)

if __name__ == "__main__":
    main()
