# SeldonianML

Code for reproducing the results from the ICLR 2022 paper, "Fairness Guarantees under Demographic Shift".

# Installation

First, install Python 3.x, Numpy (1.16+), and Cython (0.29+).

The remaining dependencies can be installed by executing the following command from the Python directory of : 

	pip install -r requirements.txt

# Usage

The experiments from the paper can be executed by running the provided batch file from the Python directory, as follows:

     ./experiments/scripts/iclr_ds_experiments.bat
     
Once the experiments complete, the figures found in the paper can be generated using the following two commands, 

     python -m experiments.scripts.iclr_figures_adult
     python -m experiments.scripts.iclr_figures_adult --unknown_ds
     python -m experiments.scripts.iclr_figures_brazil
     python -m experiments.scripts.iclr_figures_brazil --unknown_ds
    
Once completed, the new figures will be saved to `Python/figures/*` by default.

# License

SeldonianML is released under the MIT license.
