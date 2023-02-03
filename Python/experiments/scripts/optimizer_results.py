import pandas as pd
import os
import numpy as np
from scipy import stats
import math
import jinja2

pd.options.mode.chained_assignment = None

PATH = "results/output"

FC_MAP = {
    "di": "Disparate Impact",
    "dp": "Demographic Parity",
    }

MODEL_MAP = {
    'QSRC': 'Shifty',
}

def aggragate_table(cma, bfgs):
    cma_df = cma.loc[(cma["n_train"]==60000) & (cma["name"]=="QSRC")]
    bfgs_df = bfgs.loc[(bfgs["n_train"]==60000) & (bfgs["name"]=="QSRC")]

    cma_df["name"] = "cma"
    bfgs_df["name"] = "bfgs"

    df = pd.concat([cma_df, bfgs_df])
    df = df[["name", "original_nsf", "antagonist_acc_mean", "antagonist_failed_mean", "runtime"]].reset_index(drop=True)
    df.set_index("name", inplace=True)
    df.index.name = None

    return df

if __name__ == "__main__":
    # Read all json files in the output folder
    results = {}
    for file in os.listdir(PATH):
        results[file[5:-5]]=pd.read_json(os.path.join(PATH, file), orient="records")

    # Extract data and generate tables
    for dataset in ["adult", "brazil"]:
        for fc in ["di", "dp"]:

            fixed = {'cma':     results[f"{dataset}_fixed_ds_rl_{fc}"],
                     'bfgs':    results[f"{dataset}_fixed_ds_rl_{fc}_bfgs"]}
            antag = {'cma':     results[f"{dataset}_antag_ds_rl_{fc}"],
                     'bfgs':    results[f"{dataset}_antag_ds_rl_{fc}_bfgs"]}

            fixed_df = aggragate_table(fixed['cma'], fixed['bfgs'])
            antag_df = aggragate_table(antag['cma'], antag['bfgs'])

            combined = pd.concat({"Fixed DS": fixed_df, "Unknown DS": antag_df}, axis=1)
            combined.rename(columns={"original_nsf": "NSF", "antagonist_acc_mean": "Acc", "antagonist_failed_mean": "FR"}, inplace=True)
            latex = combined.style.format(
                na_rep="n/a", precision=3
                ).highlight_max(
                    props="font-weight:bold", subset=[("Fixed DS", "Acc"), ("Unknown DS", "Acc")]
                    ).highlight_min(
                        props="font-weight:bold", subset=[("Fixed DS", "FR"), ("Unknown DS", "FR")]
                        ).to_latex(
                            label=f"{fc}_{dataset}_bounds", caption=f"{FC_MAP[fc]} - {dataset} dataset", position="H", hrules=True, convert_css=True, multicol_align="c", position_float="centering") # header = ["NSF", "acc orig", "acc depl"]
            latex = latex.replace("\midrule", "\cmidrule(r){2-5} \cmidrule{6-9}")
            # latex = latex.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\begin{threeparttable}")
            # latex = latex.replace("\end{table}", "\end{threeparttable}\n\end{table}")
            # footnote = get_footnote(fixed_stats, antag_stats)
            # latex = latex.replace("\end{threeparttable}", footnote + "\n\end{threeparttable}")
            print(latex, "\n")