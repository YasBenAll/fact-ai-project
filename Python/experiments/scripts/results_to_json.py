import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()

if __name__ == '__main__':

    parser.add_argument("--directory", type=str, default="results")
    parser.add_argument("--bounds", type=bool, default=False)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.directory, "output"), exist_ok=True)
    results = os.listdir(args.directory)

    for i, result in enumerate(results):
        if os.path.isdir(os.path.join(args.directory, result)):
            # print(os.listdir(os.path.join("results", result)))
            for filename in os.listdir(os.path.join(args.directory, result)):
                if filename.endswith(".h5"):
                    if "incomplete" not in filename:
                        name = filename[:-3]
                        filename = os.path.join(args.directory, result, filename)
                        print(filename)

                        params = pd.read_hdf(filename, key='task_parameters')
                        results = pd.read_hdf(filename, key="results") # If this throws "KeyError: 'No object named results in the file'", your code didn't run properly

                        # extract acceptance rate
                        joint = results.merge(params, how="left", left_on="tid", right_index=True)
                        NSF = joint[["name", "n_train", "original_nsf"]].groupby(by=["name", "n_train"]).mean().reset_index()
                        NSF_se = joint[["name", "n_train", "original_nsf"]].groupby(by=["name", "n_train"]).sem().reset_index()


                        # extract total runtimes
                        runtimes = joint[["name", "n_train", "runtime"]].groupby(by=["name", "n_train"]).sum().reset_index()

                        # extract means and standard errors
                        solution_found = joint.loc[joint.original_nsf == False]
                        solution_found["original_failed"] = solution_found["original_g"] > 0
                        solution_found["antagonist_failed"] = solution_found["antagonist_g"] > 0
                        means = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_failed", "antagonist_failed"]].groupby(by=["name", "n_train"]).mean(numeric_only=False).add_suffix("_mean").reset_index()
                        ses = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_failed", "antagonist_failed"]].groupby(by=["name", "n_train"]).sem(numeric_only=False).add_suffix("_se").reset_index()
                        std = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_failed", "antagonist_failed"]].groupby(by=["name", "n_train"]).std(numeric_only=False).add_suffix("_std").reset_index()
                        counts = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_failed", "antagonist_failed"]].groupby(by=["name", "n_train"]).count().add_suffix("_count").reset_index()
                        # merge
                        aggregated = NSF.merge(means, how="left", on=["name", "n_train"]).merge(ses, how="left", on=["name", "n_train"]).merge(std, how="left", on=["name", "n_train"]).merge(counts, how="left", on=["name", "n_train"]).merge(runtimes, how="left", on=["name", "n_train"])
                        if "interpolation_factor" in params.columns:
                            aggregated["interpolation_factor"] = params["interpolation_factor"]
                        fill = ""
                        if args.bounds:
                            fill = str(i).zfill(2)
                        aggregated["original_nsf_se"] = NSF_se["original_nsf"]
                        aggregated.to_json(os.path.join(args.directory, "output", name + fill) + ".json", indent=4, orient="records")