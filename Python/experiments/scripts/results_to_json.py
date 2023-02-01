import json
import argparse
import pandas as pd
import os

directory = "results"

results = os.listdir(directory)

if __name__ == '__main__':
    for result in results:
        if os.path.isdir(os.path.join(directory, result)):
            # print(os.listdir(os.path.join("results", result)))
            for filename in os.listdir(os.path.join(directory, result)):
                if filename.endswith(".h5"):
                    if "incomplete" not in filename:
                        name = filename[:-3]
                        filename = os.path.join(directory, result, filename)
                        print(filename)

                        params = pd.read_hdf(filename, key='task_parameters')
                        results = pd.read_hdf(filename, key="results") # If this throws "KeyError: 'No object named results in the file'", your code didn't run properly

                        # extract acceptance rate
                        joint = results.merge(params, how="left", left_on="tid", right_index=True)
                        NSF = joint[["name", "n_train", "original_nsf"]].groupby(by=["name", "n_train"]).mean().reset_index()

                        # extract means and standard errors
                        solution_found = joint.loc[joint.original_nsf == False]
                        solution_found["original_failed"] = solution_found["original_g"] > 0
                        solution_found["antagonist_failed"] = solution_found["antagonist_g"] > 0
                        means = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_failed", "antagonist_failed"]].groupby(by=["name", "n_train"]).mean(numeric_only=False).add_suffix("_mean").reset_index()
                        ses = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_failed", "antagonist_failed"]].groupby(by=["name", "n_train"]).sem(numeric_only=False).add_suffix("_se").reset_index()
                        std = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_failed", "antagonist_failed"]].groupby(by=["name", "n_train"]).std(numeric_only=False).add_suffix("_std").reset_index()
                        counts = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_failed", "antagonist_failed"]].groupby(by=["name", "n_train"]).count().add_suffix("_count").reset_index()
                        # merge
                        aggregated = NSF.merge(means, how="left", on=["name", "n_train"]).merge(ses, how="left", on=["name", "n_train"]).merge(std, how="left", on=["name", "n_train"]).merge(counts, how="left", on=["name", "n_train"])
                        if "interpolation_factor" in params.columns:
                            aggregated["interpolation_factor"] = params["interpolation_factor"]
                        aggregated.to_json(os.path.join("results", "output", name) + ".json", indent=4, orient="records")