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
    cma_df = cma.loc[cma["name"]=="QSRC"]
    bfgs_df = bfgs.loc[bfgs["name"]=="QSRC"]
    cma_df["name"] = "CMA-ES"
    bfgs_df["name"] = "BFGS"

    df = pd.concat([cma_df, bfgs_df])
    df_new = df.loc[df["n_train"] == 60000][["name", "original_nsf", "antagonist_acc_mean", "antagonist_failed_mean"]].reset_index(drop=True)

    df_new["Delta Acc"] = df_new["antagonist_acc_mean"].reset_index(drop=True) - df.loc[df["n_train"] == 10000]["antagonist_acc_mean"].reset_index(drop=True)

    df_new.set_index("name", inplace=True)
    df_new.index.name = None

    return df_new

if __name__ == "__main__":
    
    os.makedirs("tables", exist_ok=True)

    # Read all json files in the output folder
    results = {}
    for file in os.listdir(PATH):
        results[file[5:-5]]=pd.read_json(os.path.join(PATH, file), orient="records")

    all_tables = ""

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

            all_tables += latex
            all_tables += "\n"
            

    text_file = open("tables/optimizers.txt", "w")
    text_file.write(all_tables)
    text_file.close()