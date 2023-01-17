import numpy as np
import matplotlib
import os
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import hex2color

import utils
from utils import keyboard
from utils.io import SMLAResultsReader
from utils.argsweep    import ArgumentSweeper
from datasets import brazil_with_race as brazil
import matplotlib.gridspec as gridspec

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

if __name__ == '__main__':
	with ArgumentSweeper() as parser:
		parser.add_argument('--unknown_ds',  action='store_true', help='Generates results for the unknown_ds experiments.')
		args = parser.parse_args()

	# Whether to save figures. If false, figures are displayed only.
	save_figs = True

	# Location to save the figures (will be created if nonexistent)
	mode = 'antag' if args.unknown_ds else 'fixed'
	figpath = f'figures/iclr_{mode}_demographic_shift_brazil_rl'

	# Figure format
	fmt = 'pdf'
	if fmt == 'pdf':
		matplotlib.rc('pdf', fonttype=42)

	# Figure DPI for raster formats
	dpi = 200

	# Paths to results files. Figures will be skipped if data cannot be found.
	# Note that the legend is based off of the EO figure, so it will not be 
	# generated if EO data is unavailable.
	di_path     = f'results/iclr_brazil_{mode}_ds_rl_di_0/iclr_brazil_{mode}_ds_rl_di.h5'
	dp_path     = f'results/iclr_brazil_{mode}_ds_rl_dp_0/iclr_brazil_{mode}_ds_rl_dp.h5'
	eodds_path  = f'results/iclr_brazil_{mode}_ds_rl_eodds_0/iclr_brazil_{mode}_ds_rl_eodds.h5'
	pe_path     = f'results/iclr_brazil_{mode}_ds_rl_pe_0/iclr_brazil_{mode}_ds_rl_pe.h5'
	eo_path     = f'results/iclr_brazil_{mode}_ds_rl_eopp_0/iclr_brazil_{mode}_ds_rl_eopp.h5'

	def thousands_fmt(x, pos):
		return f'{x/10**3:,.0f}K'
	def percentage_fmt(x, pos):
		return f'{x:,.0f}\%'
	ThousandsFmt = mtick.FuncFormatter(thousands_fmt)
	PercentageFmt = mtick.FuncFormatter(percentage_fmt)

	# Epsilon constants used in experiments
	di_e     =  -0.80
	dp_e     =  0.15 
	eodds_e  =  0.35 
	pe_e     =  0.2 
	eo_e     =  0.2

	# Value of delta used in experiments
	delta = 0.05

	# Constants for rendering figures
	n_total = brazil.load(gpa_cutoff=3.0).training_splits()['X'].shape[0]

	# Mapping from model names that will appear on legends
	pprint_map = {
		'SC'              : 'Seldonian',
		'QSC'             : 'Quasi-Seldonian',
		'SRC'             : 'Seldonian$_{GDS}$',
		'QSRC'            : 'Shifty',
		'FairlearnSVC'    : 'Fairlearn',
		'LinSVC'          : 'Baseline', #'SVC$_{linear}$',
		'SGD' 	          : 'Baseline', #'SGD',
		'SGD(hinge)'      : 'SGD$_{hinge}$',
		'SGD(log)' 	      : 'SGD$_{log}$',
		'SGD(perceptron)' : 'SGD$_{perc}$',
		'SVC' 	          : 'SVC$_{rbf}$',
		'FairConst'       : 'Fairness Constraints',
		'FairRobust'      : 'RFLearn'
	}
	legend_priority = {
	'Seldonian':0,
	'Quasi-Seldonian':-0.5,
	'Seldonian$_{GDS}$':1,
	'Shifty':0.5,
	'Baseline':-1,
	'Baseline':-1,
	'Fairness Constraints':-0.7,
	'Fairlearn':-0.71,
	'RFLearn':-0.75,
	}
	standard_smla_names = ['SC', 'QSC']
	robust_smla_names   = ['SRC', 'QSRC']

	keep_mname_list = ['SC','QSC','QSRC','FairConst','FairlearnSVC','FairRobust']

	# Create the figure directory if nonexistent
	if save_figs and not(os.path.isdir(figpath)):
		os.makedirs(figpath)		



	#############
	#  Helpers  #
	#############

	def save(fig, path, *args, **kwargs):
		if not(os.path.isdir(figpath)):
			os.makedirs(figpath)
		path = os.path.join(figpath, path)
		print('Saving figure to \'%s\'' % path)
		fig.savefig(path, *args, **kwargs)

	def get_ls(name):
		# if name == 'SC' or name == 'SRC':
		# 	return '--'
		# elif name == 'QSC' or name == 'QSRC':
		# 	return '-'
		if name == 'QSC':
			return '--'
		return '-'

	def get_lw(name):
		if name == 'SRC' or name == 'QSRC':
			return 2
		return 1

	def get_brazil_stats(path, include_fairlearn=False, include_fairconst=False, keep_mname_list=keep_mname_list):
		''' Helper for extracting resutls from brazil results files. '''
		results_container = SMLAResultsReader(path)
		results_container.open()

		task_parameters = results_container._store['task_parameters']
		results = results_container._store['results']

		n_train = np.array(task_parameters.n_train)
		arates, arates_se = [], [] # Acceptance rates and SEs
		ofrates, ofrates_se = [], [] # Failure rates ans SEs (rate that accepted solutions have g(theta) > 0 on the test set)
		dfrates, dfrates_se = [], [] # Failure rates ans SEs (rate that accepted solutions have g(theta) > 0 on the test set)
		olrates, olrates_se = [], [] # Test set error and SEs
		dlrates, dlrates_se = [], [] # Test set error and SEs
		mnames = np.unique(results.name).astype(str)
		mnames = [mn for mn in mnames if mn in keep_mname_list]
		pmnames = np.array([ pprint_map[name] for name in mnames ])
		for tid, _ in enumerate(n_train):
			_arates, _arates_se = [], []
			_ofrates, _ofrates_se = [], []
			_dfrates, _dfrates_se = [], []
			_olrates, _olrates_se = [], []
			_dlrates, _dlrates_se = [], []
			for mname in mnames:
				_results = results[np.logical_and(results.tid==tid, results.name==mname)]			
				if len(np.unique(_results.pid)) > 1:
					_results = _results[_results.pid==0]

				accepts = ~np.array(_results.original_nsf)
				# keyboard()
				original_gs = np.array([ _g for _g in _results.original_g])
				antagonist_gs = np.array([ _g for _g in _results.antagonist_g])
				og = np.array([ (g>0 if a else False) for (g,a) in zip(original_gs,accepts) ])
				dg = np.array([ (g>0 if a else False) for (g,a) in zip(antagonist_gs,accepts) ])
				oa = np.array(_results.original_acc)
				da = np.array(_results.antagonist_acc)
				n = len(accepts)
				n_accepts = sum(accepts)
				
				# Record I[ a(D) != NSF ]
				_arates.append(np.mean(accepts))
				_arates_se.append(np.std(accepts,ddof=1)/np.sqrt(len(accepts)))

				# Estimate Pr( g(a(D)) > 0 )
				_ofrates.append(og.mean())
				_dfrates.append(dg.mean())
				if n > 1:
					_ofrates_se.append(og.std(ddof=1)/np.sqrt(n))
					_dfrates_se.append(dg.std(ddof=1)/np.sqrt(n))
				else:
					_ofrates_se.append(np.nan)
					_dfrates_se.append(np.nan)

				# Estimate E[ loss | a(D) != NSF ]
				if n_accepts > 1:
					_olrates.append(oa[accepts].mean())
					_dlrates.append(da[accepts].mean())
					_olrates_se.append(oa[accepts].std(ddof=1)/np.sqrt(n_accepts))
					_dlrates_se.append(da[accepts].std(ddof=1)/np.sqrt(n_accepts))
				elif n_accepts == 1:
					_olrates.append(oa[accepts].mean())
					_dlrates.append(da[accepts].mean())
					_olrates_se.append(np.nan)
					_dlrates_se.append(np.nan)
				else:
					_olrates.append(np.nan)
					_dlrates.append(np.nan)
					_olrates_se.append(np.nan)
					_dlrates_se.append(np.nan)

			arates.append(_arates)
			ofrates.append(_ofrates)
			dfrates.append(_dfrates)
			olrates.append(_olrates)
			dlrates.append(_dlrates)
			arates_se.append(_arates_se)
			ofrates_se.append(_ofrates_se)
			dfrates_se.append(_dfrates_se)
			olrates_se.append(_olrates_se)
			dlrates_se.append(_dlrates_se)

		arates    = np.array(arates)
		arates_se = np.array(arates_se)
		ofrates    = np.array(ofrates)
		ofrates_se = np.array(ofrates_se)
		dfrates    = np.array(dfrates)
		dfrates_se = np.array(dfrates_se)
		olrates    = np.array(olrates)
		olrates_se = np.array(olrates_se)
		dlrates    = np.array(dlrates)
		dlrates_se = np.array(dlrates_se)
		results_container.close()

		# Assign colors to each method
		# This part is a hack to get reasonable colors for each method. If more methods are
		#   added this section should be changed.
		colors = []
		for nm in mnames:
			if nm in robust_smla_names:
				colors.append(hex2color('#4daf4a'))
			elif nm in standard_smla_names:
				colors.append(hex2color('#377eb8'))
			elif nm == 'FairlearnSVC':
				colors.append(hex2color('#FF9E44'))
			elif nm == 'FairConst':
				colors.append(hex2color('#e41a1c'))
			elif nm == 'FairRobust':
				colors.append(hex2color('#6f32a8'))
			else:
				colors.append(hex2color('#e41a1c'))
		def add_noise(X, e=0.01):
			return X + e*(np.random.random(size=X.shape) - 0.5)
		out = {
			'mnames'    : mnames,
			'pmnames'   : pmnames,
			'colors'    : colors,
			'n_train'     : n_train,
			'arate_v_n'    : arates,
			'arate_se_v_n' : arates_se,
			'ofrate_v_n'    : add_noise(ofrates, 0.0),
			'ofrate_se_v_n' : ofrates_se,
			'dfrate_v_n'    : add_noise(dfrates, 0.0),
			'dfrate_se_v_n' : dfrates_se,
			'olrate_v_n'    : 100 * olrates,
			'olrate_se_v_n' : 100 * olrates_se,
			'dlrate_v_n'    : 100 * dlrates,
			'dlrate_se_v_n' : 100 * dlrates_se
		}
		return out


out = get_brazil_stats("results\iclr_brazil_fixed_ds_rl_pe_0\iclr_brazil_fixed_ds_rl_pe.h5")


print(out)