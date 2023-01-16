import numpy as np
import warnings
from time import time
import pandas as pd
from itertools import count, product
from copy import deepcopy
from scipy import optimize

from sklearn import linear_model as LM
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

from baselines.fairlearn.classred import expgrad
from baselines.fairlearn import moments
import baselines.fair_classification.utils as fc_ut
import baselines.fair_classification.loss_funcs as fc_lf
from baselines.fair_robust.unlabeled import UnlabeledFairRobust

# SeldonianML imports
from datasets     import adult as adult
from core.base.sc import SeldonianClassifier
from utils.argsweep    import ArgumentSweeper
from utils.rvs         import ConstraintManager, get_parser, get_classification_cm
from utils.constraints import make_constraints
from utils.experiments import launcher
from utils.experiments import demographic_shift as ds
import utils

import sys


# Supress sklearn FutureWarnings for SGD
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


########################
#   Model Evaluators   #
########################

# Wrapper for training each model and evaluating it under antagonistic resampling

def _evaluate_model(dataset, trainf, mp):
	cm = get_classification_cm(mp['constraints'])

	# Resample the base dataset uniformly to obtain a training dataset
	n_train = dataset.resample_n_train 
	n_candidate = np.floor(mp['r_cand_v_safe'] * n_train).astype(int)
	n_safety   = n_train - n_candidate
	dataset0 = dataset.resample(n_candidate=n_candidate, n_safety=n_safety, n_test=0)
	dataset0._tparams = dataset._tparams
	dataset0._seed = dataset._seed
	t = time()
	predictf, is_nsf = trainf(dataset0, mp)
	t = time()-t
	dshift_opts = {	k:mp[k] for k in ['demographic_variable', 'demographic_variable_values', 'demographic_marginals','known_demographic_terms']}
	acc_orig, g_orig, acc_ant, g_ant = ds.evaluate_antagonistic_demographic_shift(predictf, mp['constraints'], dataset, dshift_opts)
	
	return {
		'original_nsf'   : is_nsf,
		'original_acc'   : acc_orig,
		'original_g'     : g_orig,
		'antagonist_acc' : acc_ant,
		'antagonist_g'   : g_ant,
		'runtime'        : t
	}

# Methods to train models

def _get_fairlearn(dataset, mp):
	# Train the model
	# Load the dataset and convert it to a pandas dataframe
	split = dataset.training_splits()
	Xt, Yt, Tt = split['X'], split['Y'], split['R']
	Xt = pd.DataFrame(Xt)
	# Convert Y to be in {0,1} instead of {-1,1} for compatibility with fairlearn
	Yt[Yt==-1] = 0
	Yt = pd.Series(Yt)
	Tt = pd.Series(Tt)
	# Use expgrad with a linear SVC

	# Note that this fairlearn implementation only supports DemographicParity and EqualOpportunity
	# When other definitions are requested, we enforce DP or EO based on which is most reasonable
	defs = {
		'demographicparity'  : moments.DP,
		'disparateimpact'    : moments.EO,
		'equalizedodds'      : moments.EO,
		'equalopportunity'   : moments.EO,
		'predictiveequality' : moments.EO }
	cons = defs[mp['definition'].lower()]()
	
	# Train fairlearn using expgrad with a linear SVC
	base_model = LinearSVC(loss=mp['loss'], penalty=mp['penalty'], fit_intercept=mp['fit_intercept'])
	try:
		results, hs = expgrad(Xt, Tt, Yt, base_model, cons=cons, eps=mp['fl_e'])
	except ValueError as e:
		s = '\nexpgrad failed.\n    shapes:\n            X: %r\n            Y: %r\n            T: %r\n   tparams: %r\n      seed:%d\n   mparams: %r\n' % (split['X'].shape, split['Y'].shape, split['S'].shape, dataset._tparams, dataset._seed, mp)
		print(s)
		sys.stdout.flush()
		raise(e)


	def predictf(X, results=results):
		Yp = np.array(np.round(results.best_classifier(X)))
		try:
			Yp[Yp==0] = -1
		except TypeError:
			Yp = Yp if Yp == 1 else -1
		return Yp
	return predictf, False

def _get_fair_constraints(dataset, mp):
	# FairConstraints is constructed to simultaneously enforce disparate impact and disparate treatment,
	# thus the training process is the same regardless of the actual definition we're evaluating.	
	# Configure the constraints and weights
	apply_fairness_constraints = 1
	apply_accuracy_constraint  = 0
	sep_constraint = 0
	gamma = None
	e = -mp['e']*100
	# Train the model using the cov that produced the smallest p >= e
	split = dataset.training_splits()
	X, Y, S, R = split['X'], split['Y'], split['S'], split['R']
	x_control = {'S':S.astype(np.int64), 'R':R.astype(np.int64)}
	sensitive_attrs = ['S', 'R']
	sensitive_attrs_to_cov_thresh = {'S':0.1, 
									 'R':0.01}
	w = fc_ut.train_model(X, Y, x_control, fc_lf._logistic_loss, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
	def predictf(_X):
		Yp = np.sign(np.dot(_X, w))
		try:
			Yp[Yp==0] = -1
		except:
			pass
		return Yp
	return predictf, False

def _get_hoeff_sc(dataset, mp, enforce_robustness=False):
	model_params = {
		'verbose'     : False,
		'shape_error' : True,
		'model_type'  : mp['model_type'],
		'ci_mode'     : 'hoeffding',
		'cs_scale'    : mp['cs_scale'],
		'robust_loss' : False}
	if enforce_robustness:
		for k in ['demographic_variable', 'demographic_variable_values', 'demographic_marginals','known_demographic_terms' ,'robust_loss']:
			model_params[k] = mp[k]

	# Train SC using hoeffding's inequality
	apply_fairness_constraints = 1
	apply_accuracy_constraint  = 0
	sep_constraint = 0
	gamma = None
	e = -mp['e']*100

	split = dataset.training_splits()
	X, Y, S, R = split['X'], split['Y'], split['S'], split['R']
	x_control = {'S':S.astype(np.int64), 'R':R.astype(np.int64)}
	sensitive_attrs = ['S', 'R']
	sensitive_attrs_to_cov_thresh = {'S':0.1, 
									 'R':0.01}
	w = fc_ut.train_model(X, Y, x_control, fc_lf._logistic_loss, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

	model = SeldonianClassifier(mp['constraints'], mp['deltas'], **model_params)
	accept = model.fit(dataset, n_iters=mp['n_iters'], optimizer_name=mp['optimizer'], theta0=w)
	return model.predict, ~accept

def _get_ttest_sc(dataset, mp, enforce_robustness=False):
	model_params = {
		'verbose'     : False,
		'shape_error' : True,
		'model_type'  : mp['model_type'],
		'ci_mode'     : 'ttest',
		'cs_scale'    : mp['cs_scale'],
		'robust_loss' : False}
	if enforce_robustness:
		for k in ['demographic_variable', 'demographic_variable_values', 'demographic_marginals','known_demographic_terms', 'robust_loss']:
			model_params[k] = mp[k]

	# Train SC using hoeffding's inequality
	apply_fairness_constraints = 1
	apply_accuracy_constraint  = 0
	sep_constraint = 0
	gamma = None
	e = -mp['e']*100

	split = dataset.training_splits()
	X, Y, S, R = split['X'], split['Y'], split['S'], split['R']
	x_control = {'S':S.astype(np.int64), 'R':R.astype(np.int64)}
	sensitive_attrs = ['S', 'R']
	sensitive_attrs_to_cov_thresh = {'S':0.1, 
									 'R':0.01}
	w = fc_ut.train_model(X, Y, x_control, fc_lf._logistic_loss, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
	model = SeldonianClassifier(mp['constraints'], mp['deltas'], **model_params)
	accept = model.fit(dataset, n_iters=mp['n_iters'], optimizer_name=mp['optimizer'], theta0=w)
	return model.predict, ~accept

def _get_sgd(dataset, mp):
	split = dataset.training_splits()
	Xt, Yt = split['X'], split['Y']
	if mp['loss']=='log':
		model = LM.LogisticRegression(fit_intercept=mp['fit_intercept'])
	else:
		model = LM.SGDClassifier(loss=mp['loss'], penalty=mp['penalty'], fit_intercept=mp['fit_intercept'], max_iter=1000, alpha=0.000001)
	model.fit(Xt, Yt)
	return model.predict, False

def _get_svc(dataset, mp):
	split = dataset.training_splits()
	Xt, Yt = split['X'], split['Y']
	model = SVC(gamma=mp['gamma'], C=mp['C'], kernel=mp['kernel'])
	model.fit(Xt, Yt)
	return model.predict, False

def _get_linsvc(dataset, mp):
	split = dataset.training_splits()
	Xt, Yt = split['X'], split['Y']
	model = LinearSVC(loss=mp['loss'], penalty=mp['penalty'], fit_intercept=mp['fit_intercept'])
	model.fit(Xt, Yt)
	return model.predict, False

def _get_fair_robust(dataset, mp):
	split = dataset.training_splits()
	Xt, Yt, St = split['X'], split['Y'], split['R']
	Yt = 1.0*(Yt == 1)
	model = UnlabeledFairRobust()
	model.fit(Xt, Yt, St)
	def predictf(X, model=model):
		Yp = model.predict(X)
		return 1*(Yp==1) - 1*(Yp==0)
	return predictf, False

# Actual evaluation functions

def eval_fairlearn(dataset, mp):
	return _evaluate_model(dataset, _get_fairlearn, mp)

def eval_fair_constraints(dataset, mp):
	return _evaluate_model(dataset, _get_fair_constraints, mp)

def eval_hoeff_sc(dataset, mp):
	trainf = lambda dataset, mp: _get_hoeff_sc(dataset, mp, enforce_robustness=False)
	return _evaluate_model(dataset, trainf, mp)

def eval_hoeff_sc_robust(dataset, mp):
	trainf = lambda dataset, mp: _get_hoeff_sc(dataset, mp, enforce_robustness=True)
	return _evaluate_model(dataset, trainf, mp)

def eval_ttest_sc(dataset, mp):
	trainf = lambda dataset, mp: _get_ttest_sc(dataset, mp, enforce_robustness=False)
	return _evaluate_model(dataset, trainf, mp)

def eval_ttest_sc_robust(dataset, mp):
	trainf = lambda dataset, mp: _get_ttest_sc(dataset, mp, enforce_robustness=True)
	return _evaluate_model(dataset, trainf, mp)

def eval_sgd(dataset, mp):
	return _evaluate_model(dataset, _get_sgd, mp)

def eval_svc(dataset, mp):
	return _evaluate_model(dataset, _get_svc, mp)

def eval_linsvc(dataset, mp):
	return _evaluate_model(dataset, _get_linsvc, mp)

def eval_fair_robust(dataset, mp):
	return _evaluate_model(dataset, _get_fair_robust, mp)


######################
#   Dataset Loader   #
######################

def load_dataset(tparams, seed):
	dset_args = {
		'r_train'     : 1.0, 
		'include_intercept' : True,
		'include_R'   : tparams['include_R'], 
		'include_S'   : tparams['include_S'], 
		'use_pct'     : 1.0,
		'seed'        : seed,
		'standardize' : tparams['standardize'],
		'R0' : 'Black',
		'R1' : 'White'
	}
	dataset = adult.load(**dset_args)	
	dataset.resample_n_train = tparams['n_train']
	dataset._tparams = deepcopy(tparams)
	dataset._seed = seed
	return dataset


############
#   Main   #
############

if __name__ == '__main__':

	# Note: This script computes experiments for the cross product of all values given for the
	#       sweepable arguments. 
	# Note: Sweepable arguments allow inputs of the form, <start>:<end>:<increment>, which are then
	#       expanded into ranges via np.arange(<start>, <end>, <increment>). 
	with ArgumentSweeper() as parser:
		parser.add_argument('base_path', type=str)
		parser.add_argument('--include_R',  action='store_true',      help='Whether or not to include race as a predictive feature.')
		parser.add_argument('--include_S',  action='store_true',      help='Whether or not to include sex as a predictive feature.')
		parser.add_argument('--standardize',  action='store_true', help='Whether or not to standardize input features.')
		parser.add_argument('--n_jobs',     type=int,   default=4,    help='Number of processes to use.')
		parser.add_argument('--n_trials',   type=int,   default=10,   help='Number of trials to run.')
		parser.add_argument('--n_iters',    type=int,   default=10,   help='Number of SMLA training iterations.')
		parser.add_argument('--optimizer',  type=str,   default='cmaes', help='Choice of optimizer to use.')
		parser.add_argument('--definition', type=str,   default='DisparateImpact',        help='Choice of safety definition to enforce.')
		parser.add_argument('--e',          type=float,  default=0.05, help='Value for epsilon.')
		parser.add_argument('--d',          type=float,  default=0.05, help='Value for delta.')
		parser.add_argument('--robust_loss',  action='store_true',      help='Causes the loss function to estimate post-demographic shift loss.')
		parser.add_sweepable_argument('--n_train',   type=int,  default=10000,   nargs='*', help='Number of samples to draw from the population for training.')
		parser.add_sweepable_argument('--r_train_v_test', type=float, default=0.4,  nargs='*', help='Ratio of data used for training vs testing.')
		parser.add_argument('--r_cand_v_safe',  type=float, default=0.4,  help='Ratio of training data used for candidate selection vs safety checking. (SMLA only)')
		parser.add_sweepable_argument('--model_type',     type=str, default='linear', nargs='*', help='Base model type to use for SMLAs.')
		parser.add_argument('--fixed_dist',  action='store_true',      help='Fixed the distribution post-deployment (only works when dshift_var=race.')
		parser.add_argument('--dshift_var', type=str,       default='race', help='Choice of variable to evaluate demographic shift for.')
		parser.add_argument('--dshift_alpha', type=float,   default=0.1,    help='Width of intervals around true marginals representing valid demographic shifts.')
		parser.add_argument('--cs_scale', type=float, default=1.0,  help='Scaling factor for predicted confidence intervals during candidate selection.')
		args = parser.parse_args()
		args_dict = dict(args.__dict__)

		# Generate thje constraints and deltas		
		population  = adult.load(R0='Black', R1='White')
		if args.dshift_var.lower()[0] == 's':
			constraints = make_constraints(args.definition, 'R', np.unique(population._R), args.e)
		if args.dshift_var.lower()[0] == 'r':
			constraints = make_constraints(args.definition, 'S', np.unique(population._S), args.e)
		deltas = [ args.d for _ in constraints ]


		print()
		print(args.definition,':')
		print('   Interpreting constraint string \'%s\''  % constraints[0])
		print('                               as \'%r\'.' % get_parser().parse(constraints[0]))

		
		smla_names = ['SC', 'QSC', 'SRC', 'QSRC']
		# model_evaluators = {
		# 	# 'SC'           : eval_hoeff_sc,
		# 	# 'QSC'          : eval_ttest_sc,
		# 	# 'SRC'          : eval_hoeff_sc_robust,
		# 	# 'QSRC'         : eval_ttest_sc_robust,
		# 	# # 'SGD'          : eval_sgd,
		# 	# # 'LinSVC'       : eval_linsvc,
		# 	# # 'SVC'          : eval_svc
		# 	# 'FairConst'    : eval_fair_constraints,
		# 	'FairlearnSVC' : eval_fairlearn
		# 	# 'FairRobust'   : eval_fair_robust
		# }
		model_evaluators = {
			'SC'           : eval_hoeff_sc,
			'QSC'          : eval_ttest_sc,
			'SRC'          : eval_hoeff_sc_robust,
			'QSRC'         : eval_ttest_sc_robust,
			# 'SGD'          : eval_sgd,
			# 'LinSVC'       : eval_linsvc,
			# 'SVC'          : eval_svc
			'FairConst'    : eval_fair_constraints,
			'FairlearnSVC' : eval_fairlearn,
			'FairRobust'   : eval_fair_robust
		}



		#    Store task parameters:
		tparams = {k:args_dict[k] for k in ['n_jobs', 'base_path', 'r_train_v_test', 'include_R', 'include_S', 'standardize', 'n_train']}

		# Generate options for enforcing robustness constraints
		if args.dshift_var.lower() == 'sex':
			D = get_parser(mode='inner').parse('S')
			D_values = population._S
		elif args.dshift_var.lower() == 'race':
			D = get_parser(mode='inner').parse('R')
			D_values = population._R
		else:
			raise RuntimeError('This experiment does not support demographic shift for the variable \'%s\'' % args.dshift_var)
		unique_D_values = np.unique(D_values)
		Pr_D = np.array([ (D_values==d).mean() for d in unique_D_values ])
		assert (args.dshift_alpha >= 0) and (args.dshift_alpha <= 1.0), 'Demographic shift alpha value must be between 0 and 1.'
		if args.fixed_dist:
			smla_dshift_opts = {
				'demographic_variable'        : D,
				'demographic_variable_values' : unique_D_values,
				# 'demographic_marginals'       : np.array([0.25, 0.75]),
				'demographic_marginals'       : np.array([0.15, 0.85]),
				'known_demographic_terms'     : ds.get_population_conditionals(population.all_sets(), constraints, D)
			}
		else:
			smla_dshift_opts = {
				'demographic_variable'        : D,
				'demographic_variable_values' : unique_D_values,
				'demographic_marginals'       : ds.make_intervals(Pr_D, args.dshift_alpha, epsilon=1e-3),
				'known_demographic_terms'     : ds.get_population_conditionals(population.all_sets(), constraints, D)
			}

		# Fill in parameter dictionaries for each model
		srl_mparam_names  = ['n_iters','optimizer','model_type', 'definition', 'e', 'cs_scale', 'robust_loss']
		bsln_mparam_names = ['definition', 'e']
		mparams = {}
		for name in model_evaluators.keys():
			if name in smla_names:
				mparams[name] = {k:args_dict[k] for k in srl_mparam_names}
			else:
				mparams[name] = {k:args_dict[k] for k in bsln_mparam_names}
			mparams[name]['constraints'] = constraints
			mparams[name]['deltas'] = deltas
			mparams[name]['dshift_alpha'] = args.dshift_alpha
			mparams[name]['dshift_var']   = args.dshift_var
			mparams[name]['r_cand_v_safe'] = args.r_cand_v_safe
			mparams[name].update(smla_dshift_opts)
		# mparams['SGD'].update(loss=['hinge','log','perceptron'], penalty='l2', fit_intercept=False)
		# mparams['SVC'].update(kernel=['rbf'], gamma=2, C=1)
		# mparams['LinSVC'].update(loss=['hinge'], penalty='l2', fit_intercept=False)
		mparams['FairConst'].update(cov=[0.01])
		mparams['FairlearnSVC'].update(loss=['hinge'], penalty='l2', fit_intercept=False, fl_e=[0.01, 0.1])
		
		#    Expand the parameter sets into a set of configurations
		args_to_expand = parser._sweep_argnames + ['loss', 'kernel', 'cov', 'fl_e', 'n_train']
		tparams, mparams = launcher.make_parameters(tparams, mparams, expand=args_to_expand)	

		print(ds.make_intervals(Pr_D, args.dshift_alpha, epsilon=1e-3))
		print(ds.make_intervals(Pr_D, args.dshift_alpha, epsilon=1e-3))


		print()
		# Create a results file and directory
		save_path = launcher.prepare_paths(args.base_path, tparams, mparams, smla_names, root='results', filename=None)
		print()
		# Run the experiment
		launcher.run(args.n_trials, save_path, model_evaluators, load_dataset, tparams, mparams, n_workers=args.n_jobs, seed=None)


		# tp = tparams[-1]
		# dataset = load_dataset(tp, 0)
		# for i,mp in enumerate(mparams['FairlearnSVC']):
		# 	print(f'mp[{i}]:')

		# 	mp['disable_linprog'] = False
		# 	results = model_evaluators['FairlearnSVC'](dataset, mp)
		# 	klen = max([ len(k) for k in results.keys() ])
		# 	print('   Disable linprog = False:')
		# 	for key, value in results.items():
		# 		print(f'      {k.rjust(klen)}: {value}')

		# 	mp['disable_linprog'] = True
		# 	results = model_evaluators['FairlearnSVC'](dataset, mp)
		# 	klen = max([ len(k) for k in results.keys() ])
		# 	print('Disable linprog = False:')
		# 	for key, value in results.items():
		# 		print(f'      {k.rjust(klen)}: {value}')

		# for name, evalf in model_evaluators.items():
		# 	print(name)
		# 	mp = mparams[name][0]
		# 	results = evalf(dataset, mp)
		# 	print(results)