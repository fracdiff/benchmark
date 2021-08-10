import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

GITHUB_GRAY = "#6a737d"
GITHUB_BLUE = "#0366d6"
GITHUB_ORANGE = "#f66a0a"
CYCLE = matplotlib.cycler(color=[GITHUB_BLUE, GITHUB_ORANGE])

seaborn.set_style("whitegrid")

FONTSIZE = 18
matplotlib.rcParams["axes.labelsize"] = FONTSIZE
matplotlib.rcParams["axes.titlesize"] = FONTSIZE
matplotlib.rcParams["figure.titlesize"] = FONTSIZE
matplotlib.rcParams["legend.fontsize"] = FONTSIZE
matplotlib.rcParams["xtick.labelsize"] = FONTSIZE
matplotlib.rcParams["ytick.labelsize"] = FONTSIZE
matplotlib.rcParams["axes.prop_cycle"] = CYCLE
matplotlib.rcParams["grid.color"] = GITHUB_GRAY
matplotlib.rcParams["axes.edgecolor"] = GITHUB_GRAY
matplotlib.rcParams["legend.edgecolor"] = GITHUB_GRAY
matplotlib.rcParams["axes.linewidth"] = 1.6
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["figure.figsize"] = (10, 5)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.1


if __name__ == "__main__":
    df_all = pd.read_csv("out/results.csv")

    # Create markdown table
    def fmt(x: float) -> str:
        return f"{x:.3f}"

    df = df_all[df_all["n_features"] == 1].copy()
    series_time_mean = (1000 * df["time_mean"]).map(fmt)
    series_time_std = (1000 * df["time_std"]).map(fmt)
    df["time (ms)"] = series_time_mean.str.cat(series_time_std, sep=" +- ")
    with open("out/table_n_samples.md", "w") as f:
        md = df.pivot("n_samples", columns=["method"], values="time (ms)").to_markdown()
        f.write(md)
    print("table saved to", "out/table_n_samples.md")

    df = df_all[df_all["n_samples"] == 1000].copy()
    series_time_mean = (1000 * df["time_mean"]).map(fmt)
    series_time_std = (1000 * df["time_std"]).map(fmt)
    df["time (ms)"] = series_time_mean.str.cat(series_time_std, sep=" +- ")
    with open("out/table_n_features.md", "w") as f:
        md = df.pivot(
            "n_features", columns=["method"], values="time (ms)"
        ).to_markdown()
        f.write(md)
    print("table saved to", "out/table_n_features.md")

    # Plot

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 2, 1)
    df = df_all[df_all["n_features"] == 1]
    dff = df[df["method"] == "fracdiff"]
    dfo = df[df["method"] == "official"]
    time_mean_ms_f = 1000 * dff["time_mean"]
    time_std_ms_f = 1000 * dff["time_std"]
    time_mean_ms_o = 1000 * dfo["time_mean"]
    time_std_ms_o = 1000 * dfo["time_std"]
    plt.errorbar(
        dff["n_samples"], time_mean_ms_f, yerr=time_std_ms_f, label="fracdiff", fmt="o"
    )
    plt.errorbar(
        dfo["n_samples"], time_mean_ms_o, yerr=time_std_ms_o, label="official", fmt="o"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Samples")
    plt.ylabel("Computation Time [ms]")
    plt.legend()
    plt.title("Computation Time")

    plt.subplot(1, 2, 2)
    df = df_all[df_all["n_samples"] == 1000]
    dff = df[df["method"] == "fracdiff"]
    dfo = df[df["method"] == "official"]
    time_mean_ms_f = 1000 * dff["time_mean"]
    time_std_ms_f = 1000 * dff["time_std"]
    time_mean_ms_o = 1000 * dfo["time_mean"]
    time_std_ms_o = 1000 * dfo["time_std"]
    plt.errorbar(
        dff["n_features"], time_mean_ms_f, yerr=time_std_ms_f, label="fracdiff", fmt="o"
    )
    plt.errorbar(
        dfo["n_features"], time_mean_ms_o, yerr=time_std_ms_o, label="official", fmt="o"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Features")
    plt.ylabel("Computation Time [ms]")
    plt.legend()
    plt.title("Computation Time")

    plt.savefig("out/time.png", bbox_inches="tight")
    print("plot saved to", "out/time.png")
