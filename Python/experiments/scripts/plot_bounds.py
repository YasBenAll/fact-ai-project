import pandas as pd
import os
import matplotlib.pyplot as plt

directory = "results/output"
os.makedirs("figures", exist_ok=True)

if __name__ == '__main__':
    results = os.listdir(directory)

    dfs = []

    for result in results:
        df = pd.read_json(os.path.join(directory, result), orient="records")
        df["dataset"] = result[5]
        df["fairness_definition"] = result[-9:-7]
        dfs.append(df)

    bounds = pd.concat(dfs)

    adult_di = bounds.loc[(bounds.dataset == "a") & (bounds.fairness_definition == "di")]
    adult_dp = bounds.loc[(bounds.dataset == "a") & (bounds.fairness_definition == "dp")]
    brazil_di = bounds.loc[(bounds.dataset == "b") & (bounds.fairness_definition == "di")]
    brazil_dp = bounds.loc[(bounds.dataset == "b") & (bounds.fairness_definition == "dp")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3))

    ax1.plot(adult_di["interpolation_factor"], adult_di["antagonist_acc_mean"], label="Disparate Impact, adult dataset", color="C0")
    ax1.fill_between(
        adult_di["interpolation_factor"],
        adult_di["antagonist_acc_mean"]-adult_di["antagonist_acc_se"], 
        adult_di["antagonist_acc_mean"]+adult_di["antagonist_acc_se"],
        alpha=0.2,
        color="C0")
    ax1.plot(adult_dp["interpolation_factor"], adult_dp["antagonist_acc_mean"], label="Demographic Parity, adult dataset", color="C1")
    ax1.fill_between(
        adult_dp["interpolation_factor"],
        adult_dp["antagonist_acc_mean"]-adult_dp["antagonist_acc_se"], 
        adult_dp["antagonist_acc_mean"]+adult_dp["antagonist_acc_se"],
        alpha=0.2,
        color="C1")
    ax1.plot(brazil_di["interpolation_factor"], brazil_di["antagonist_acc_mean"], label="Disparate Impact, brazil dataset", linestyle="--", color="C0")
    ax1.fill_between(
        brazil_di["interpolation_factor"],
        brazil_di["antagonist_acc_mean"]-brazil_di["antagonist_acc_se"], 
        brazil_di["antagonist_acc_mean"]+brazil_di["antagonist_acc_se"],
        alpha=0.2,
        color="C0")
    ax1.plot(brazil_dp["interpolation_factor"], brazil_dp["antagonist_acc_mean"], label="Demographic Parity, brazil dataset", linestyle="--", color="C1")
    ax1.fill_between(
        brazil_dp["interpolation_factor"],
        brazil_dp["antagonist_acc_mean"]-brazil_dp["antagonist_acc_se"], 
        brazil_dp["antagonist_acc_mean"]+brazil_dp["antagonist_acc_se"],
        alpha=0.2,
        color="C1")
    ax1.set_title("Performance depending on bound size")
    ax1.set_xlabel("Bound size")
    ax1.set_ylabel("Classification accuracy")
    ax1.spines[['right', 'top']].set_visible(False)

    ax2.plot(adult_di["interpolation_factor"], adult_di["original_nsf"], color = "C0")
    ax2.fill_between(
        adult_di["interpolation_factor"],
        adult_di["original_nsf"]-adult_di["original_nsf_se"], 
        adult_di["original_nsf"]+adult_di["original_nsf_se"],
        alpha=0.2,
        color="C0")
    ax2.plot(adult_dp["interpolation_factor"], adult_dp["original_nsf"], color = "C1")
    ax2.fill_between(
        adult_dp["interpolation_factor"],
        adult_dp["original_nsf"]-adult_dp["original_nsf_se"], 
        adult_dp["original_nsf"]+adult_dp["original_nsf_se"],
        alpha=0.2,
        color="C1")
    ax2.plot(brazil_di["interpolation_factor"], brazil_di["original_nsf"], linestyle="--", color="C0")
    ax2.fill_between(
        brazil_di["interpolation_factor"],
        brazil_di["original_nsf"]-brazil_di["original_nsf_se"], 
        brazil_di["original_nsf"]+brazil_di["original_nsf_se"],
        alpha=0.2,
        color="C0")
    ax2.plot(brazil_dp["interpolation_factor"], brazil_dp["original_nsf"], linestyle="--", color="C1")
    ax2.fill_between(
        brazil_dp["interpolation_factor"],
        brazil_dp["original_nsf"]-brazil_dp["original_nsf_se"], 
        brazil_dp["original_nsf"]+brazil_dp["original_nsf_se"],
        alpha=0.2,
        color="C1")
    ax2.set_title("Proportion NSF depending on bound size")
    ax2.set_xlabel("Bound size")
    ax2.set_ylabel("Proportion NSF")
    ax2.spines[['right', 'top']].set_visible(False)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol = 4)
    plt.savefig("Bounds.png", facecolor='white', transparent=False, bbox_inches='tight')
    plt.show()