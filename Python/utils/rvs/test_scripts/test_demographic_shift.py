from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from core.base.sc import SeldonianClassifier
from datasets.brazil_with_race import load
from datasets.utils import get_population_conditionals, make_intervals
from utils.rvs import ConstraintManager
from utils.rvs.parser import get_parser
from utils.constraints import make_constraints



def sample(predictf, alpha, population, constraints, deltas):
	R = population._R
	unique_races = np.unique(R)
	race_var = get_parser(mode='inner').parse('R')
	Pr_R = np.array([ (R==r).mean() for r in unique_races ])
	race_constraint_opts = {
		'demographic_variable'        : race_var,
		'demographic_variable_values' : unique_races,
		'demographic_marginals'       : make_intervals(Pr_R, alpha, epsilon=1e-3),
		'known_demographic_terms'     : get_population_conditionals(population.all_sets(), constraints, race_var)
	}			
	hoeff_race_SC = SeldonianClassifier(constraints, deltas, ci_mode='hoeffding', **race_constraint_opts)
	bs_race_SC    = SeldonianClassifier(constraints, deltas, ci_mode='bootstrap', **race_constraint_opts)

	S = population._S
	unique_sexes = np.unique(S)
	sex_var = get_parser(mode='inner').parse('S')
	Pr_S = np.array([ (S==s).mean() for s in unique_sexes ])
	sex_constraint_opts = {
		'demographic_variable'        : sex_var,
		'demographic_variable_values' : unique_sexes,
		'demographic_marginals'       : make_intervals(Pr_S, alpha, epsilon=1e-3),
		'known_demographic_terms'     : get_population_conditionals(population.all_sets(), constraints, sex_var)
	}
	hoeff_sex_SC  = SeldonianClassifier(constraints, deltas, ci_mode='hoeffding', **sex_constraint_opts)
	bs_sex_SC     = SeldonianClassifier(constraints, deltas, ci_mode='bootstrap', **sex_constraint_opts)
	
	return {
		'hoeff_race' : hoeff_race_SC.safety_test(predictf, split),
		'hoeff_sex'  : hoeff_sex_SC.safety_test(predictf, split),
		'bs_race'    : bs_race_SC.safety_test(predictf, split),
		'bs_sex'     : bs_sex_SC.safety_test(predictf, split)
	}


if __name__ == '__main__':
	# Load the dataset
	population = load()

	# Generate the constraints and set the corresponding deltas
	constraints = make_constraints('DemographicParity', 'R', np.unique(population._R), 0.1) + \
				  make_constraints('DemographicParity', 'S', np.unique(population._S), 0.1)
	deltas = [ 0.05 for _ in constraints ]

	# Train a model
	split = population.training_splits()
	model = LogisticRegression(solver='lbfgs')
	model.fit(split['X'], split['Y'])
	predictf = model.predict

	# Compute upper bounds for various interval inflation factors
	alphas = np.linspace(0,1,10)
	output = defaultdict(lambda: [])
	for i,a in enumerate(alphas):
		s = sample(predictf,a,population,constraints,deltas)
		for k, v in s.items():
			output[k].append(v)
	output = { k : np.array(v) for k,v in output.items() }

	# Plot the results
	fig, axes = plt.subplots(1,2)
	fig2, axes2 = plt.subplots(1,2)
	colors = [ plt.cm.tab20(i) for i in range(len(constraints)) ]
	handles = []
	for color, constraint, hoeff, bs in zip(colors, constraints, output['hoeff_race'].T, output['bs_race'].T):
		h, = axes[0].plot(alphas, hoeff, label=constraint, c=color, ls='-')
		axes[0].plot(alphas, bs,   c=color, ls='--')
		axes2[0].plot(alphas, hoeff-bs, c=color, ls='-', label=constraint)
		handles.append(h)
	for color, constraint, hoeff, bs in zip(colors, constraints, output['hoeff_sex'].T, output['bs_sex'].T):
		axes[1].plot(alphas, hoeff, c=color, ls='-')
		axes[1].plot(alphas, bs,    c=color, ls='--')
		axes2[1].plot(alphas, hoeff-bs, c=color, ls='-', label=constraint)
	for ax in axes:
		ax.set_xlim((0,1))
	for ax in axes2:
		ax.set_xlim((0,1))
		ax.axhline(0,0,1,c='k',ls=':')
	fig.legend(handles, constraints, loc='lower center', ncol=len(constraints)//3+len(constraints)%3, mode='expand')
	fig2.legend(handles, constraints, loc='lower center', ncol=len(constraints)//3+len(constraints)%3, mode='expand')
	fig.show()
	fig2.show()