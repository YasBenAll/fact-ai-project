import numpy as np
import pandas as pd
import os.path

from datasets.dataset import ClassificationDataset
from utils.dataset import standardized, with_intercept, with_feature

BASE_URL = os.path.join('datasets', 'adult', 'adult_norm.csv')

def load(r_train=0.4, r_candidate=0.2, seed=None, include_intercept=True, use_pct=1.0, include_R=False, include_S=False, standardize=False, R0=None, R1=None, shuffle=True):
	""" Load the Adult dataset from the UCI repository. 
	
	Parameters:
		r_train (float): The proportion of the dataset to use for training.
		r_candidate (float): The proportion of the training set to use for candidate training.
		seed (int): The random seed to use.
		include_intercept (bool): Whether to include an intercept term in the dataset.
		use_pct (float): The proportion of the dataset to use.
		include_R (bool): Whether to include the protected attribute in the dataset.
		include_S (bool): Whether to include the sensitive attribute in the dataset.
		standardize (bool): Whether to standardize the dataset.
		R0 (str): The name of the first protected attribute to include in the dataset.
		R1 (str): The name of the second protected attribute to include in the dataset.
		shuffle (bool): Whether to shuffle the dataset.

	Returns:
		(dict): A dictionary containing the following keys:
			'X' (np.ndarray): The feature matrix.
			'Y' (np.ndarray): The label vector.
			'S' (np.ndarray): The sensitive attribute vector.
			'R' (np.ndarray): The protected attribute vector.
				"""
	meta_information = {
		'standardized' 		: standardize,
		'include_R'    		: include_R, 
		'include_S'    		: include_S,
		'include_intercept' : include_intercept
	}
	if not(R0 is None and R1 is None):
		meta_information.update({'R0':R0, 'R1':R1})

	random = np.random.RandomState(seed)
	
	with open(BASE_URL, 'r') as f:
		raw = list(f)
	labels, *raw = [ d.strip().split(',') for d in raw ]
	data = { k:np.array(v).astype(float) for k,v in zip(labels, np.array(raw).T)}


	race_keys = [ k for k in labels if k.startswith('race:') ]
	race_labels = [ k.split(':is_')[1] for k in race_keys ]
	race_encodings = np.array([ data[k] for k in race_keys ]).T
	race_codes = [ np.where(s)[0] for s in race_encodings ]
	I = np.array([ len(c)==1 for c in race_codes ]) # for some reason, some records have no race associated with them 
	
	data = {k:v[I] for k,v in data.items()}
	R = np.array([ race_codes[i][0] for i,keep in enumerate(I) if keep ])
	S = data['gender:is_Male'].astype(int)
	Y = data['income:is_>50K']
	Y[Y==0] = -1

	feature_keys = [ k for k in labels if not(k.startswith('race:')) and not(k in ['gender:is_Male', 'income:is_>50K', 'capital-gain', 'capital-loss', 'hours-per-week', 'age', 'educational-num'])]
	
	X = np.array([ data[k] for k in feature_keys ]).T

	meta_information.update({
			'race_codes':race_labels,
			'sex_codes':['female','male'],
			'feature_labels':feature_keys
		})

	# Reduce the dataset size as needed
	n_keep = int(np.ceil(len(X) * use_pct))
	I = np.arange(len(X))
	if shuffle:
		random.shuffle(I)
	I = I[:n_keep]
	X = X[I]
	Y = Y[I].flatten()
	S = S[I].flatten()
	R = R[I].flatten()

	# Filter out samples for races if R0 and/or R1 is specified
	if not(R1 is None and R0 is None):
		race_labels = [ l.lower() for l in race_labels ]
		if not(R0 is None) and not(R1 is None):
			i0 = race_labels.index(R0.lower())
			i1 = race_labels.index(R1.lower())
			I = np.logical_or(R==i0,R==i1)
			X = X[I]
			Y = Y[I]
			S = S[I]
			R = R[I]
			R = 0*(R==i0) + 1*(R==i1)
		elif not(R0 is None):
			i0 = race_labels.index(R0.lower())
			R = 0*(R==i0) + 1*(R!=i0)
		elif not(R1 is None):
			i1 = race_labels.index(R1.lower())
			R = 0*(R!=i1) + 1*(R==i1)

	# Compute split sizes
	n_samples   = len(X)
	n_train     = int(r_train*n_samples)
	n_test      = n_samples - n_train
	n_candidate = int(r_candidate*n_train)
	n_safety    = n_train - n_candidate


	if standardize:
		X = standardized(X)
	if include_R:
		X = with_feature(X,R)
		meta_information['feature_labels'].append('race')
	if include_S:
		X = with_feature(X,S)
		meta_information['feature_labels'].append('sex')
	if include_intercept:
		X = with_intercept(X)
		meta_information['feature_labels'].append('intercept')

	contents = { 'X':X, 'Y':Y, 'R':R, 'S':S }
	all_labels = [0, 1]
	return ClassificationDataset(all_labels, n_candidate, n_safety, n_test, seed=seed, meta_information=meta_information, **contents)
