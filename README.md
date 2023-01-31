# On the reproducibility of ”Fairness Guarantees under Demographic Shift”

This repository contains the code used for the paper [On the reproducibility of ”Fairness Guarantees under Demographic Shift”](https://arxiv.org/abs/2030.12345). It is largely similar to the original codebase of the ICLR 2022 paper [Fairness Guarantees under Demographic Shift](https://openreview.net/forum?id=wbPObLm6ueA), but contains the following contributions:

- An env.yaml file containing the correct requirements for reproducing the experiments.
- Adjustments made to the code to run it without errors on all machines (Linux, Windows, and Mac)
- Result aggregation and saving to json for ease of inspection

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

After cloning this repository, simply run:

```setup
conda env create -f Python/env.yaml
conda activate fact
```

The datasets are already contained in this repository.

## Training

The experiments from the paper can be executed by running the provided file from the Python directory, as follows:

```train
./experiments/scripts/iclr_ds_experiments.sh
```

Hyperparameters and models are specified in this file.

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

Once the experiments complete, the figures found in the paper can be generated using the following commands: 


```eval
python -m experiments.scripts.iclr_figures_adult
python -m experiments.scripts.iclr_figures_adult --unknown_ds
python -m experiments.scripts.iclr_figures_brazil
python -m experiments.scripts.iclr_figures_brazil --unknown_ds
```
    
Once completed, the new figures will be saved to `Python/figures/*` by default.

To easily inspect the raw results, the results saved to h5 files can be aggregated and converted to json using:

```json
python experiments/scripts/results_to_json.py --filename results/directory/file.h5
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

The model designed by the authors of the reproduced paper achieves the following performance on the given datasets:

| Dataset name       | Accuracy without demographic shift  | Accuracy with demographic shift |
| ------------------ | ----------------------------------- | ------------------------------- |
| [Adult](https://archive.ics.uci.edu/ml/datasets/adult)            |                                   80.2% |                                78.8% |
| [Brazil](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O35FW8)           |                                   61.3% |                                62.4% |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## License

This repository is released under the MIT license.