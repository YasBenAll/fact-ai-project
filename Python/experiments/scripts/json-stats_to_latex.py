import pandas as pd
import os
import numpy as np
from scipy import stats
import math

pd.options.mode.chained_assignment = None

NUM_TTESTS = 14
path = "results/output"

fc_map = {
    "di": "Disparate Impact",
    "dp": "Demographic Parity",
    "eodds": "Equalized Odds",
    "eopp": "Equal Opportunity",
    "pe": "Predictive Equality"
    }

model_map = {
    'SC'              : 'SC',
    'QSC'             : 'Quasi-SC',
    'QSRC'            : 'Shifty',
    'FairlearnSVC'    : 'Fairlearn',
    'FairConst'       : 'FairConst',
    'FairRobust'      : 'RFLearn'
}

def preprocess(df):
    df = df.loc[df["name"] != "SRC"]
    df["name"]=df["name"].map(model_map)
    return df

def aggragate_table(df):
    df_large = df.loc[df["n_train"] == 60000][["name", "original_nsf", "antagonist_acc_mean", "antagonist_failed_mean"]].reset_index(drop=True)
    df_large["Delta Acc"] = df_large["antagonist_acc_mean"].reset_index(drop=True) - df.loc[df["n_train"] == 10000]["antagonist_acc_mean"].reset_index(drop=True)
    df_large.loc[df_large["name"].isin(["Fairlearn", "FairConst", "RFLearn"]), "original_nsf"] = None
    df_large.set_index("name", inplace=True)
    df_large.index.name = None
    return df_large

def t_test(x1,x2, s1,s2,n1,n2):
    t = (x1-x2)/np.sqrt(s1**2/n1 + s2**2/n2)
    return abs(t)

def p_value(t, df):
    p = 1 - stats.t.cdf(t, df) # one-sided
    return p * NUM_TTESTS

def get_stats(df):
    df_shifty = df.loc[(df["name"] =="Shifty")&(df["n_train"]==60000)]
    df_max = df[(df["name"] !="Shifty")&(df["n_train"]==60000)]
    idmax = df_max.sort_values(by=["antagonist_acc_mean"], ascending=False)["antagonist_acc_mean"].idxmax()
    df_max = df_max.loc[idmax]
    # print(df_max["name"])

    n1 = df_shifty["antagonist_acc_count"].item()
    if n1 < 3:
        return -1, None, None

    n2 = df_max["antagonist_acc_count"].item()

    x1 = df_shifty["antagonist_acc_mean"].item()
    x2 = df_max["antagonist_acc_mean"].item()

    s1 = df_shifty["antagonist_acc_std"].item()
    s2 = df_max["antagonist_acc_std"].item()

    doff = n1+n2-2
    if math.isnan(doff):
        return -1, None, None
    doff = int(doff)

    t = t_test(x1,x2, s1,s2,n1,n2)
    p = 2*(1-stats.t.cdf(abs(t), doff))

    return p, t, doff

def get_footnote(fixed_stats, antag_stats):

    if fixed_stats[0] == -1:
        footnote1 = "\\item[1] insufficient number of solutions to perform t-test"
    elif fixed_stats[0] < 0.001:
        footnote1 = "\\item[1] significantly worse than best model, $p<0.001$, $t={:.3f}$, $df={}$".format(fixed_stats[1], fixed_stats[2])
    elif fixed_stats[0] < 0.05:
        footnote1 = "\\item[1] significantly worse than best model, $p={:.3f}$, $t={:.3f}$, $df={}$".format(fixed_stats[0], fixed_stats[1], fixed_stats[2])
    else:
        footnote1 = "\\item[1] not significantly different from best model, $p={:.3f}$, $t={:.3f}$, $df={}$".format(fixed_stats[0], fixed_stats[1], fixed_stats[2])

    if antag_stats[0] == -1:
        footnote2 = "\\item[2] insufficient number of solutions to perform t-test"
    elif antag_stats[0] < 0.001:
        footnote2 = "\\item[2] significantly worse than best model, $p<0.001$, $t={:.3f}$, $df={}$".format(antag_stats[1], antag_stats[2])
    elif antag_stats[0] < 0.05:
        footnote2 = "\\item[2] significantly worse than best model, $p={:.3f}$, $t={:.3f}$, $df={}$".format(antag_stats[0], antag_stats[1], antag_stats[2])
    else:
        footnote2 = "\\item[2] not significantly different from best model, $p={:.3f}$, $t={:.3f}$, $df={}$".format(antag_stats[0], antag_stats[1], antag_stats[2])

    footnote = "\\begin{tablenotes}\n" + footnote1 + "\n" + footnote2 + "\n\\end{tablenotes}"

    return footnote


if __name__ == "__main__":

    # Read all json files in the output folder
    results = {}
    for file in os.listdir(path):
        results[file[:-5]]=pd.read_json(os.path.join(path,file), orient="records")


    for dataset in ["adult", "brazil"]:
        for fc in ["di", "dp", "eodds", "eopp", "pe"]:
            fixed = results[f"iclr_{dataset}_fixed_ds_rl_{fc}"]
            antag = results[f"iclr_{dataset}_antag_ds_rl_{fc}"]
            fixed = preprocess(fixed)
            antag = preprocess(antag)
            fixed_stats = get_stats(fixed)
            antag_stats = get_stats(antag)
            fixed = aggragate_table(fixed)
            antag = aggragate_table(antag)
            combined = pd.concat({"Fixed DS": fixed, "Unknown DS": antag}, axis=1)
            combined.rename(columns={"original_nsf": "NSF", "antagonist_acc_mean": "Acc", "antagonist_failed_mean": "FR"}, inplace=True)
            latex = combined.style.format(
                na_rep="n/a", precision=3
                ).highlight_max(
                    props="font-weight:bold", subset=[("Fixed DS", "Acc"), ("Unknown DS", "Acc")]
                    ).highlight_min(
                        props="font-weight:bold", subset=[("Fixed DS", "FR"), ("Unknown DS", "FR")]
                        ).to_latex(
                            label=f"{fc}_{dataset}", caption=f"{fc_map[fc]} - {dataset} dataset", position="H", hrules=True, convert_css=True, multicol_align="c", position_float = "centering") # header = ["NSF", "acc orig", "acc depl"]
            latex = latex.replace("\midrule", "\cmidrule(r){2-5} \cmidrule{6-9}")
            latex = latex.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\begin{threeparttable}")
            latex = latex.replace("\end{table}", "\end{threeparttable}\n\end{table}")
            footnote = get_footnote(fixed_stats, antag_stats)
            latex = latex.replace("\end{threeparttable}", footnote + "\n\end{threeparttable}")
            print(latex)
            print()

