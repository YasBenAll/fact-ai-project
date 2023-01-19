import json
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    params = pd.read_hdf(args.filename, key='task_parameters')
    results = pd.read_hdf(args.filename, key='results') # If this throws "KeyError: 'No object named results in the file'", your code didn't run properly

    # extract acceptance rate
    joint = results.merge(params, how="left", left_on="tid", right_index=True)
    NSF = joint[["name", "n_train", "original_nsf"]].groupby(by=["name", "n_train"]).mean().reset_index()

    # extract means and standard errors
    solution_found = joint.loc[joint.original_nsf == False]
    means = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_g", "antagonist_g"]].groupby(by=["name", "n_train"]).mean().add_suffix("_mean").reset_index()
    ses = solution_found[["name", "n_train", "original_acc", "antagonist_acc", "original_g", "antagonist_g"]].groupby(by=["name", "n_train"]).sem().add_suffix("_se").reset_index()

    # merge
    aggregated = NSF.merge(means, how="left", on=["name", "n_train"]).merge(ses, how="left", on=["name", "n_train"])
    aggregated.to_json(args.filename[:-2] + "json", indent=4, orient="records")