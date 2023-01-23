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
from datasets import adult
import matplotlib.gridspec as gridspec
from utils.argsweep    import ArgumentSweeper

import pandas as pd

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
	figpath = f'figures/iclr_{mode}_demographic_shift_adult_rl'

	# Figure format
	fmt = 'png'
	if fmt == 'pdf':
		matplotlib.rc('pdf', fonttype=42)

	# Figure DPI for raster formats
	dpi = 200

	# Paths to results files. Figures will be skipped if data cannot be found.
	# Note that the legend is based off of the EO figure, so it will not be 
	# generated if EO data is unavailable.
	di_path     = f'results/iclr_adult_{mode}_ds_rl_di_0/iclr_adult_{mode}_ds_rl_di.h5'
	dp_path     = f'results/iclr_adult_{mode}_ds_rl_dp_0/iclr_adult_{mode}_ds_rl_dp.h5'
	eodds_path  = f'results/iclr_adult_{mode}_ds_rl_eodds_0/iclr_adult_{mode}_ds_rl_eodds.h5'
	pe_path     = f'results/iclr_adult_{mode}_ds_rl_pe_0/iclr_adult_{mode}_ds_rl_pe.h5'
	eo_path     = f'results/iclr_adult_{mode}_ds_rl_eopp_0/iclr_adult_{mode}_ds_rl_eopp.h5'
	all_paths = {
		'di':di_path,
		'dp':dp_path,
		'eodds':eodds_path,
		'pe':pe_path,
		'eo':eo_path
	}

	def get(path):
		results_container = SMLAResultsReader(path)
		results_container.open()
		task_parameters = results_container._store['task_parameters']
		return results_container._store['results']
	def test(results):
		names = np.unique(results.name)
		for name in names:
			m_results = results[results.name==name]
			print('%s:' % name)
			for pid in np.unique(m_results.pid):
				print('  pid %d:' % pid)
				for tid in np.unique(m_results.tid):
					print('    tid %d: %d' % (tid, len(m_results[np.logical_and(m_results.tid==tid, m_results.pid==pid)])))
	for mn,path in all_paths.items():
		print()
		print(mn)
		test(get(path))

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
	n_total = adult.load(R0='Black', R1='White').training_splits()['X'].shape[0]

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
	'RFLearn':0.1,
	}
	standard_smla_names = ['SC', 'QSC']
	robust_smla_names   = ['SRC', 'QSRC']

	keep_mname_list = ['SC','QSC','QSRC','FairConst','FairlearnSVC','FairRobust']
	# keep_mname_list = ['QSRC','FairRobust']

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

	def get_adult_stats(path, include_fairlearn=False, include_fairconst=False, keep_mname_list=keep_mname_list):
		''' Helper for extracting resutls from adult results files. '''
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
		print('-'*100)
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
				if mname == 'FairRobust':
					print(mname)
					print(dg)
				
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
				# print(mname)
				# print('OA', _olrates)
				# print('OG', original_gs)
				# print('AA', _dlrates)
				# print('AG', antagonist_gs)

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
		# results_container.close()

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
			'ofrate_v_n'    : add_noise(ofrates, 0.02),
			'ofrate_se_v_n' : ofrates_se,
			'dfrate_v_n'    : add_noise(dfrates, 0.02),
			'dfrate_se_v_n' : dfrates_se,
			'olrate_v_n'    : 100 * olrates,
			'olrate_se_v_n' : 100 * olrates_se,
			'dlrate_v_n'    : 100 * dlrates,
			'dlrate_se_v_n' : 100 * dlrates_se
		}
		return out




	####################################################################
	# DisparateImpact: Accuracy, Acceptance Rates, and Failure Rates #
	####################################################################


	ff_acc = plt.figure(constrained_layout=False, figsize=(7, 2))
	ff_dfr = plt.figure(constrained_layout=False, figsize=(7, 2))

	gs_acc = ff_acc.add_gridspec(1, 2, hspace=1.0,wspace=0.5)
	gs_dfr = ff_acc.add_gridspec(1, 2, hspace=1.0,wspace=0.5)

	ax_dacc_di  = ff_acc.add_subplot(gs_acc[0,0])
	ax_dacc_dp  = ff_acc.add_subplot(gs_acc[0,1])
	ax_dfr_di   = ff_dfr.add_subplot(gs_dfr[0,0])
	ax_dfr_dp   = ff_dfr.add_subplot(gs_dfr[0,1])



	if not(os.path.exists(di_path)):
		print('No results found at path \'%s\'. Skipped.' % di_path)
	else:
		D = get_adult_stats(di_path)
		arates = D['arate_v_n']
		arates_se = D['arate_se_v_n']
		ofrates = D['ofrate_v_n']
		ofrates_se = D['ofrate_se_v_n']
		dfrates = D['dfrate_v_n']
		dfrates_se = D['dfrate_se_v_n']
		oacc_v_n = D['olrate_v_n']
		oacc_se_v_n = D['olrate_se_v_n']
		dacc_v_n = D['dlrate_v_n']
		dacc_se_v_n = D['dlrate_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals = D['n_train']
		
		fig = plt.figure(constrained_layout=False, figsize=(10, 4))
		gs = fig.add_gridspec(2, 6, hspace=1.0,wspace=2)

		ax_ar   = fig.add_subplot(gs[:, :2])
		ax_oacc = fig.add_subplot(gs[0, 2:4])
		ax_ofr  = fig.add_subplot(gs[0, 4:6])
		ax_dacc = fig.add_subplot(gs[1, 2:4])
		ax_dfr  = fig.add_subplot(gs[1, 4:6])
		
		# Plot acceptance rate
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, (1-ar), c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, ((1-ar)+se), ((1-ar)-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=3.5)
		ax_ar.set_ylabel('NO_SOLUTION_FOUND', labelpad=7)
		# ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.set_ylim((0,1))
		ax_ar.xaxis.set_major_formatter(ThousandsFmt)
		# ax_ar.yaxis.set_major_formatter(PercentageFmt)


		# Plot accuracy values
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(oacc_v_n.T)[::-1],(oacc_se_v_n.T)[::-1]):
			ax_oacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_oacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_oacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_oacc.set_ylabel('Accuracy\n(Original)', labelpad=7)
		# ax_oacc.set_ylim((33,67))
		# ax_oacc.set_xscale("log")
		ax_oacc.set_xlim(right=max(nvals))
		ax_oacc.yaxis.set_major_formatter(PercentageFmt)
		ax_oacc.xaxis.set_major_formatter(ThousandsFmt)


		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(dacc_v_n.T)[::-1],(dacc_se_v_n.T)[::-1]):
			ax_dacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_dacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_dacc.set_ylabel('Accuracy\n(Deployed)', labelpad=7)
		# ax_dacc.set_ylim((33,67))
		# ax_dacc.set_xscale("log")
		ax_dacc.set_xlim(right=max(nvals))
		ax_dacc.yaxis.set_major_formatter(PercentageFmt)
		ax_dacc.xaxis.set_major_formatter(ThousandsFmt)

		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(dacc_v_n.T)[::-1],(dacc_se_v_n.T)[::-1]):
			ax_dacc_di.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dacc_di.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_dacc_di.set_xlabel('Training Samples', labelpad=3.5)
		ax_dacc_di.set_ylabel('Accuracy\n(Deployed)', labelpad=7)
		# ax_dacc_di.set_ylim((33,67))
		# ax_dacc_di.set_xscale("log")
		ax_dacc_di.set_xlim(right=max(nvals))
		ax_dacc_di.yaxis.set_major_formatter(PercentageFmt)
		ax_dacc_di.xaxis.set_major_formatter(ThousandsFmt)

		# Plot failure rate
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(ofrates.T)[::-1], (ofrates_se.T)[::-1]):
			ax_ofr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ofr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_ofr.axhline(delta*100, color='k', linestyle=':')
		ax_ofr.set_xlabel('Training Samples', labelpad=3.5)
		ax_ofr.set_ylabel('Failure Rate\n(Original)', labelpad=7)
		# ax_ofr.set_xscale("log")
		ax_ofr.set_ylim((-np.nanmax(ofrates)*5, np.nanmax(ofrates)*110))
		ax_ofr.set_xlim(right=max(nvals))
		ax_ofr.yaxis.set_major_formatter(PercentageFmt)
		ax_ofr.xaxis.set_major_formatter(ThousandsFmt)

		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(dfrates.T)[::-1], (dfrates_se.T)[::-1]):
			ax_dfr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dfr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_dfr.axhline(delta*100, color='k', linestyle=':')
		ax_dfr.set_xlabel('Training Samples', labelpad=3.5)
		ax_dfr.set_ylabel('Failure Rate\n(Deployed)', labelpad=7)
		# ax_dfr.set_xscale("log")
		ax_dfr.set_ylim((-np.nanmax(dfrates)*5, np.nanmax(dfrates)*110))
		ax_dfr.set_xlim(right=max(nvals))
		ax_dfr.yaxis.set_major_formatter(PercentageFmt)
		ax_dfr.xaxis.set_major_formatter(ThousandsFmt)

		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(dfrates.T)[::-1], (dfrates_se.T)[::-1]):
			ax_dfr_di.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dfr_di.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_dfr_di.axhline(delta*100, color='k', linestyle=':')
		ax_dfr_di.set_xlabel('Training Samples', labelpad=3.5)
		ax_dfr_di.set_ylabel('Failure Rate\n(Deployed)', labelpad=7)
		# ax_dfr_di.set_xscale("log")
		ax_dfr_di.set_ylim((-np.nanmax(dfrates)*5, np.nanmax(dfrates)*110))
		ax_dfr_di.set_xlim(right=max(nvals))
		ax_dfr_di.yaxis.set_major_formatter(PercentageFmt)
		ax_dfr_di.xaxis.set_major_formatter(ThousandsFmt)

		# Finalize the figure and display/save
		for ax in [ax_ar, ax_oacc, ax_ofr, ax_dacc, ax_dfr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(top=0.93, left=0.065, right=0.98, bottom=0.21)
		if save_figs:
			save(fig,'iclr_adult_demographic_shift_sex_di.%s' % fmt, dpi=dpi)
		else:
			fig.show()



	####################################################################
	# DemographicParity: Accuracy, Acceptance Rates, and Failure Rates #
	####################################################################

	if not(os.path.exists(dp_path)):
		print('No results found at path \'%s\'. Skipped.' % dp_path)
	else:
		D = get_adult_stats(dp_path)
		arates = D['arate_v_n']
		arates_se = D['arate_se_v_n']
		ofrates = D['ofrate_v_n']
		ofrates_se = D['ofrate_se_v_n']
		dfrates = D['dfrate_v_n']
		dfrates_se = D['dfrate_se_v_n']
		oacc_v_n = D['olrate_v_n']
		oacc_se_v_n = D['olrate_se_v_n']
		dacc_v_n = D['dlrate_v_n']
		dacc_se_v_n = D['dlrate_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals = D['n_train']
		
		fig = plt.figure(constrained_layout=False, figsize=(10, 4))
		gs = fig.add_gridspec(2, 6, hspace=1.0,wspace=2)

		ax_ar   = fig.add_subplot(gs[:, :2])
		ax_oacc = fig.add_subplot(gs[0, 2:4])
		ax_ofr  = fig.add_subplot(gs[0, 4:6])
		ax_dacc = fig.add_subplot(gs[1, 2:4])
		ax_dfr  = fig.add_subplot(gs[1, 4:6])
		
		# Plot acceptance rate
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, (1-ar), c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, ((1-ar)+se), ((1-ar)-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=3.5)
		ax_ar.set_ylabel('NO_SOLUTION_FOUND', labelpad=7)
		# ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.set_ylim((0,1))
		ax_ar.xaxis.set_major_formatter(ThousandsFmt)
		# ax_ar.yaxis.set_major_formatter(PercentageFmt)

		# Plot accuracy values
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(oacc_v_n.T)[::-1],(oacc_se_v_n.T)[::-1]):
			ax_oacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_oacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_oacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_oacc.set_ylabel('Accuracy\n(Original)', labelpad=7)
		# ax_oacc.set_ylim((33,67))
		# ax_oacc.set_xscale("log")
		ax_oacc.set_xlim(right=max(nvals))
		ax_oacc.yaxis.set_major_formatter(PercentageFmt)
		ax_oacc.xaxis.set_major_formatter(ThousandsFmt)
		

		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(dacc_v_n.T)[::-1],(dacc_se_v_n.T)[::-1]):
			ax_dacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_dacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_dacc.set_ylabel('Accuracy\n(Deployed)', labelpad=7)
		# ax_dacc.set_ylim((33,67))
		# ax_dacc.set_xscale("log")
		ax_dacc.set_xlim(right=max(nvals))
		ax_dacc.yaxis.set_major_formatter(PercentageFmt)
		ax_dacc.xaxis.set_major_formatter(ThousandsFmt)
		
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(dacc_v_n.T)[::-1],(dacc_se_v_n.T)[::-1]):
			ax_dacc_dp.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dacc_dp.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_dacc_dp.set_xlabel('Training Samples', labelpad=3.5)
		ax_dacc_dp.set_ylabel('Accuracy\n(Deployed)', labelpad=7)
		# ax_dacc_dp.set_ylim((33,67))
		# ax_dacc_dp.set_xscale("log")
		ax_dacc_dp.set_xlim(right=max(nvals))
		ax_dacc_dp.yaxis.set_major_formatter(PercentageFmt)
		ax_dacc_dp.xaxis.set_major_formatter(ThousandsFmt)

		# Plot failure rate
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(ofrates.T)[::-1], (ofrates_se.T)[::-1]):
			ax_ofr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ofr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_ofr.axhline(delta*100, color='k', linestyle=':')
		ax_ofr.set_xlabel('Training Samples', labelpad=3.5)
		ax_ofr.set_ylabel('Failure Rate\n(Original)', labelpad=7)
		# ax_ofr.set_xscale("log")
		ax_ofr.set_ylim((-np.nanmax(ofrates)*5, np.nanmax(ofrates)*110))
		ax_ofr.set_xlim(right=max(nvals))
		ax_ofr.yaxis.set_major_formatter(PercentageFmt)
		ax_ofr.xaxis.set_major_formatter(ThousandsFmt)
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(dfrates.T)[::-1], (dfrates_se.T)[::-1]):
			ax_dfr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dfr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_dfr.axhline(delta*100, color='k', linestyle=':')
		ax_dfr.set_xlabel('Training Samples', labelpad=3.5)
		ax_dfr.set_ylabel('Failure Rate\n(Deployed)', labelpad=7)
		# ax_dfr.set_xscale("log")
		ax_dfr.set_ylim((-np.nanmax(dfrates)*5, np.nanmax(dfrates)*110))
		ax_dfr.set_xlim(right=max(nvals))
		ax_dfr.yaxis.set_major_formatter(PercentageFmt)
		ax_dfr.xaxis.set_major_formatter(ThousandsFmt)

		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(dfrates.T)[::-1], (dfrates_se.T)[::-1]):
			ax_dfr_dp.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dfr_dp.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_dfr_dp.axhline(delta*100, color='k', linestyle=':')
		ax_dfr_dp.set_xlabel('Training Samples', labelpad=3.5)
		ax_dfr_dp.set_ylabel('Failure Rate\n(Deployed)', labelpad=7)
		# ax_dfr_dp.set_xscale("log")
		ax_dfr_dp.set_ylim((-np.nanmax(dfrates)*5, np.nanmax(dfrates)*110))
		ax_dfr_dp.set_xlim(right=max(nvals))
		ax_dfr_dp.yaxis.set_major_formatter(PercentageFmt)
		ax_dfr_dp.xaxis.set_major_formatter(ThousandsFmt)

		# Finalize the figure and display/save
		for ax in [ax_ar, ax_oacc, ax_ofr, ax_dacc, ax_dfr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(top=0.93, left=0.065, right=0.98, bottom=0.21)
		if save_figs:
			save(fig,'iclr_adult_demographic_shift_sex_dp.%s' % fmt, dpi=dpi)
		else:
			fig.show()


	ax_dfr_di.set_title('Disparate Impact')
	ax_dfr_dp.set_title('Demographic Parity')
	ff_acc.subplots_adjust(top=0.83, left=0.165, right=0.98, bottom=0.21)
	ff_dfr.subplots_adjust(top=0.83, left=0.165, right=0.98, bottom=0.21)
	save(ff_acc, 'iclr_adult_talk_acc.%s' % fmt, dpi=dpi)
	save(ff_dfr, 'iclr_adult_talk_fr.%s' % fmt, dpi=dpi)


	# ################################################################
	# # EqualizedOdds: Accuracy, Acceptance Rates, and Failure Rates #
	# ################################################################

	if not(os.path.exists(eodds_path)):
		print('No results found at path \'%s\'. Skipped.' % eodds_path)
	else:
		D = get_adult_stats(eodds_path)
		arates = D['arate_v_n']
		arates_se = D['arate_se_v_n']
		ofrates = D['ofrate_v_n']
		ofrates_se = D['ofrate_se_v_n']
		dfrates = D['dfrate_v_n']
		dfrates_se = D['dfrate_se_v_n']
		oacc_v_n = D['olrate_v_n']
		oacc_se_v_n = D['olrate_se_v_n']
		dacc_v_n = D['dlrate_v_n']
		dacc_se_v_n = D['dlrate_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals = D['n_train']
		
		fig = plt.figure(constrained_layout=False, figsize=(10, 1.9))
		gs = fig.add_gridspec(2, 6, hspace=1.0,wspace=2)

		ax_ar   = fig.add_subplot(gs[:, :2])
		ax_oacc = fig.add_subplot(gs[0, 2:4])
		ax_ofr  = fig.add_subplot(gs[0, 4:6])
		ax_dacc = fig.add_subplot(gs[1, 2:4])
		ax_dfr  = fig.add_subplot(gs[1, 4:6])
		
		# Plot acceptance rate
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, (1-ar), c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, ((1-ar)+se), ((1-ar)-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=3.5)
		ax_ar.set_ylabel('NO_SOLUTION_FOUND', labelpad=7)
		# ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.set_ylim((0,1))
		ax_ar.xaxis.set_major_formatter(ThousandsFmt)
		# ax_ar.yaxis.set_major_formatter(PercentageFmt)

		# Plot accuracy values
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(oacc_v_n.T)[::-1],(oacc_se_v_n.T)[::-1]):
			ax_oacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_oacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_oacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_oacc.set_ylabel('Accuracy\n(Original)', labelpad=7)
		# ax_oacc.set_ylim((33,67))
		# ax_oacc.set_xscale("log")
		ax_oacc.set_xlim(right=max(nvals))
		ax_oacc.yaxis.set_major_formatter(PercentageFmt)
		ax_oacc.xaxis.set_major_formatter(ThousandsFmt)

		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(dacc_v_n.T)[::-1],(dacc_se_v_n.T)[::-1]):
			ax_dacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_dacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_dacc.set_ylabel('Accuracy\n(Deployed)', labelpad=7)
		# ax_dacc.set_ylim((33,67))
		# ax_dacc.set_xscale("log")
		ax_dacc.set_xlim(right=max(nvals))
		ax_dacc.yaxis.set_major_formatter(PercentageFmt)
		ax_dacc.xaxis.set_major_formatter(ThousandsFmt)

		# Plot failure rate
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(ofrates.T)[::-1], (ofrates_se.T)[::-1]):
			ax_ofr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ofr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_ofr.axhline(delta*100, color='k', linestyle=':')
		ax_ofr.set_xlabel('Training Samples', labelpad=3.5)
		ax_ofr.set_ylabel('Failure Rate\n(Original)', labelpad=7)
		# ax_ofr.set_xscale("log")
		ax_ofr.set_ylim((-np.nanmax(ofrates)*5, np.nanmax(ofrates)*110))
		ax_ofr.set_xlim(right=max(nvals))
		ax_ofr.yaxis.set_major_formatter(PercentageFmt)
		ax_ofr.xaxis.set_major_formatter(ThousandsFmt)

		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(dfrates.T)[::-1], (dfrates_se.T)[::-1]):
			ax_dfr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dfr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_dfr.axhline(delta*100, color='k', linestyle=':')
		ax_dfr.set_xlabel('Training Samples', labelpad=3.5)
		ax_dfr.set_ylabel('Failure Rate\n(Deployed)', labelpad=7)
		# ax_dfr.set_xscale("log")
		ax_dfr.set_ylim((-np.nanmax(dfrates)*5, np.nanmax(dfrates)*110))
		ax_dfr.set_xlim(right=max(nvals))
		ax_dfr.yaxis.set_major_formatter(PercentageFmt)
		ax_dfr.xaxis.set_major_formatter(ThousandsFmt)

		# Finalize the figure and display/save
		for ax in [ax_ar, ax_oacc, ax_ofr, ax_dacc, ax_dfr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(top=0.93, left=0.065, right=0.98, bottom=0.21)
		if save_figs:
			save(fig,'iclr_adult_demographic_shift_sex_eodds.%s' % fmt, dpi=dpi)
		else:
			fig.show()


	# ###################################################################
	# # PredictiveEquality: Accuracy, Acceptance Rates, and Failure Rates #
	# ###################################################################

	if not(os.path.exists(pe_path)):
		print('No results found at path \'%s\'. Skipped.' % pe_path)
	else:
		D = get_adult_stats(pe_path)
		arates = D['arate_v_n']
		arates_se = D['arate_se_v_n']
		ofrates = D['ofrate_v_n']
		ofrates_se = D['ofrate_se_v_n']
		dfrates = D['dfrate_v_n']
		dfrates_se = D['dfrate_se_v_n']
		oacc_v_n = D['olrate_v_n']
		oacc_se_v_n = D['olrate_se_v_n']
		dacc_v_n = D['dlrate_v_n']
		dacc_se_v_n = D['dlrate_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals = D['n_train']
		
		fig = plt.figure(constrained_layout=False, figsize=(10, 1.9))
		gs = fig.add_gridspec(2, 6, hspace=1.0,wspace=2)

		ax_ar   = fig.add_subplot(gs[:, :2])
		ax_oacc = fig.add_subplot(gs[0, 2:4])
		ax_ofr  = fig.add_subplot(gs[0, 4:6])
		ax_dacc = fig.add_subplot(gs[1, 2:4])
		ax_dfr  = fig.add_subplot(gs[1, 4:6])
		
		# Plot acceptance rate
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, (1-ar), c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, ((1-ar)+se), ((1-ar)-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=3.5)
		ax_ar.set_ylabel('NO_SOLUTION_FOUND', labelpad=7)
		# ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.set_ylim((0,1))
		ax_ar.xaxis.set_major_formatter(ThousandsFmt)
		# ax_ar.yaxis.set_major_formatter(PercentageFmt)

		# Plot accuracy values
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(oacc_v_n.T)[::-1],(oacc_se_v_n.T)[::-1]):
			ax_oacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_oacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_oacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_oacc.set_ylabel('Accuracy\n(Original)', labelpad=7)
		# ax_oacc.set_ylim((33,67))
		# ax_oacc.set_xscale("log")
		ax_oacc.set_xlim(right=max(nvals))
		ax_oacc.yaxis.set_major_formatter(PercentageFmt)
		ax_oacc.xaxis.set_major_formatter(ThousandsFmt)

		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(dacc_v_n.T)[::-1],(dacc_se_v_n.T)[::-1]):
			ax_dacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_dacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_dacc.set_ylabel('Accuracy\n(Deployed)', labelpad=7)
		# ax_dacc.set_ylim((33,67))
		# ax_dacc.set_xscale("log")
		ax_dacc.set_xlim(right=max(nvals))
		ax_dacc.yaxis.set_major_formatter(PercentageFmt)
		ax_dacc.xaxis.set_major_formatter(ThousandsFmt)

		# Plot failure rate
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(ofrates.T)[::-1], (ofrates_se.T)[::-1]):
			ax_ofr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ofr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_ofr.axhline(delta*100, color='k', linestyle=':')
		ax_ofr.set_xlabel('Training Samples', labelpad=3.5)
		ax_ofr.set_ylabel('Failure Rate\n(Original)', labelpad=7)
		# ax_ofr.set_xscale("log")
		ax_ofr.set_ylim((-np.nanmax(ofrates)*5, np.nanmax(ofrates)*110))
		ax_ofr.set_xlim(right=max(nvals))
		ax_ofr.yaxis.set_major_formatter(PercentageFmt)
		ax_ofr.xaxis.set_major_formatter(ThousandsFmt)

		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(dfrates.T)[::-1], (dfrates_se.T)[::-1]):
			ax_dfr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dfr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_dfr.axhline(delta*100, color='k', linestyle=':')
		ax_dfr.set_xlabel('Training Samples', labelpad=3.5)
		ax_dfr.set_ylabel('Failure Rate\n(Deployed)', labelpad=7)
		# ax_dfr.set_xscale("log")
		ax_dfr.set_ylim((-np.nanmax(dfrates)*5, np.nanmax(dfrates)*110))
		ax_dfr.set_xlim(right=max(nvals))
		ax_dfr.yaxis.set_major_formatter(PercentageFmt)
		ax_dfr.xaxis.set_major_formatter(ThousandsFmt)

		# Finalize the figure and display/save
		for ax in [ax_ar, ax_oacc, ax_ofr, ax_dacc, ax_dfr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(top=0.93, left=0.065, right=0.98, bottom=0.21)
		if save_figs:
			save(fig,'iclr_adult_demographic_shift_sex_pe.%s' % fmt, dpi=dpi)
		else:
			fig.show()


	# #####################################################################
	# # EqualOpportunity: Accuracy, Acceptance Rates, and Failure Rates #
	# #####################################################################


	if not(os.path.exists(eo_path)):
		print('No results found at path \'%s\'. Skipped.' % eo_path)
	else:
		D = get_adult_stats(eo_path)
		arates = D['arate_v_n']
		arates_se = D['arate_se_v_n']
		ofrates = D['ofrate_v_n']
		ofrates_se = D['ofrate_se_v_n']
		dfrates = D['dfrate_v_n']
		dfrates_se = D['dfrate_se_v_n']
		oacc_v_n = D['olrate_v_n']
		oacc_se_v_n = D['olrate_se_v_n']
		dacc_v_n = D['dlrate_v_n']
		dacc_se_v_n = D['dlrate_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals = D['n_train']
		
		fig = plt.figure(constrained_layout=False, figsize=(10, 1.9))
		gs = fig.add_gridspec(2, 6, hspace=1.0,wspace=2)

		ax_ar   = fig.add_subplot(gs[:, :2])
		ax_oacc = fig.add_subplot(gs[0, 2:4])
		ax_ofr  = fig.add_subplot(gs[0, 4:6])
		ax_dacc = fig.add_subplot(gs[1, 2:4])
		ax_dfr  = fig.add_subplot(gs[1, 4:6])
		
		# Plot acceptance rate
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, (1-ar), c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, ((1-ar)+se), ((1-ar)-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=3.5)
		ax_ar.set_ylabel('NO_SOLUTION_FOUND', labelpad=7)
		# ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.set_ylim((0,1))
		ax_ar.xaxis.set_major_formatter(ThousandsFmt)
		# ax_ar.yaxis.set_major_formatter(PercentageFmt)

		# Plot accuracy values
		legend_data, added = [], []
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(oacc_v_n.T)[::-1],(oacc_se_v_n.T)[::-1]):
			line = ax_oacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_oacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
			pmn = pprint_map[mn]
			if not(pmn in added):
				added.append(pmn)
				legend_data.append(line)
		legend_data, added = legend_data[::-1], added[::-1]
		ax_oacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_oacc.set_ylabel('Accuracy\n(Original)', labelpad=7)
		# ax_oacc.set_ylim((33,67))
		# ax_oacc.set_xscale("log")
		ax_oacc.set_xlim(right=max(nvals))
		ax_oacc.yaxis.set_major_formatter(PercentageFmt)
		ax_oacc.xaxis.set_major_formatter(ThousandsFmt)

		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(dacc_v_n.T)[::-1],(dacc_se_v_n.T)[::-1]):
			ax_dacc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dacc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_dacc.set_xlabel('Training Samples', labelpad=3.5)
		ax_dacc.set_ylabel('Accuracy\n(Deployed)', labelpad=7)
		# ax_dacc.set_ylim((33,67))
		# ax_dacc.set_xscale("log")
		ax_dacc.set_xlim(right=max(nvals))
		ax_dacc.yaxis.set_major_formatter(PercentageFmt)
		ax_dacc.xaxis.set_major_formatter(ThousandsFmt)

		# Plot failure rate
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(ofrates.T)[::-1], (ofrates_se.T)[::-1]):
			ax_ofr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ofr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_ofr.axhline(delta*100, color='k', linestyle=':')
		ax_ofr.set_xlabel('Training Samples', labelpad=3.5)
		ax_ofr.set_ylabel('Failure Rate\n(Original)', labelpad=7)
		# ax_ofr.set_xscale("log")
		ax_ofr.set_ylim((-np.nanmax(ofrates)*5, np.nanmax(ofrates)*110))
		ax_ofr.set_xlim(right=max(nvals))
		ax_ofr.yaxis.set_major_formatter(PercentageFmt)
		ax_ofr.xaxis.set_major_formatter(ThousandsFmt)

		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(dfrates.T)[::-1], (dfrates_se.T)[::-1]):
			ax_dfr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_dfr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_dfr.axhline(delta*100, color='k', linestyle=':')
		ax_dfr.set_xlabel('Training Samples', labelpad=3.5)
		ax_dfr.set_ylabel('Failure Rate\n(Deployed)', labelpad=7)
		# ax_dfr.set_xscale("log")
		ax_dfr.set_ylim((-np.nanmax(dfrates)*5, np.nanmax(dfrates)*110))
		ax_dfr.set_xlim(right=max(nvals))
		ax_dfr.yaxis.set_major_formatter(PercentageFmt)
		ax_dfr.xaxis.set_major_formatter(ThousandsFmt)

		# Finalize the figure and display/save
		for ax in [ax_ar, ax_oacc, ax_ofr, ax_dacc, ax_dfr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(top=0.93, left=0.065, right=0.98, bottom=0.21)
		if save_figs:
			save(fig,'iclr_adult_demographic_shift_sex_eo.%s' % fmt, dpi=dpi)
		else:
			fig.show()

		#####################################
		# Figure containing the legend only #
		#####################################

		fig = plt.figure(figsize=(10.75,0.3))
		priorities = [legend_priority[n] for n in added]
		added = [ added[i] for i in np.argsort(priorities)[::-1]]
		legend_data = [ legend_data[i] for i in np.argsort(priorities)[::-1]]
		fig.legend(legend_data, added, 'center', fancybox=True, ncol=len(legend_data), columnspacing=1, fontsize=11, handletextpad=0.5)
		save(fig, 'iclr_legend.%s' % fmt, dpi=dpi)
		# fig.show()