import numpy as np
import pandas as pd
import os.path

from datasets.dataset import ClassificationDataset
from utils.dataset import standardized, with_intercept, with_feature

BASE_URL = os.path.join('datasets', 'brazil', 'brazil_with_race.npz') 

def load(gpa_cutoff=3.0, r_train=0.4, r_candidate=0.2, seed=None, include_intercept=True, use_pct=1.0, include_R=False, include_S=False, standardize=False):
	"""
	Load the Brazilian dataset from the UCI repository.
	
	Parameters:
		gpa_cutoff (float): The GPA cutoff for admission to a Brazilian university.
		r_train (float): The proportion of the dataset to use for training.
		r_candidate (float): The proportion of the training set to use for candidate training.
		seed (int): The random seed to use.
		include_intercept (bool): Whether to include an intercept term in the dataset.
		use_pct (float): The proportion of the dataset to use.
		include_R (bool): Whether to include the protected attribute in the dataset.
		include_S (bool): Whether to include the sensitive attribute in the dataset.
		standardize (bool): Whether to standardize the dataset.
			
		Returns:
		A ClassificationDataset object containing the dataset.
			"""
	meta_information = {
		'standardized' 		: standardize,
		'include_R'    		: include_R, 
		'include_S'    		: include_S,
		'include_intercept' : include_intercept,
		'gpa_cutoff'		: gpa_cutoff # 3.0 is the cutoff for admission to a Brazilian university
	}

	random = np.random.RandomState(seed)
	
	D = np.load(BASE_URL)
	X = D['X'] 
	Y = (D['Y']>=gpa_cutoff).astype(float) - (D['Y']<gpa_cutoff).astype(float)
	R = D['R']
	S = D['S']


	# Reduce the dataset size as needed
	n_keep = int(np.ceil(len(X) * use_pct))
	I = np.arange(len(X))
	random.shuffle(I)
	I = I[:n_keep]
	X = X[I]
	Y = Y[I].flatten()
	S = S[I].flatten()
	R = R[I].flatten()

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
	if include_S:
		X = with_feature(X,S)
	if include_intercept:
		X = with_intercept(X)

	contents = { 'X':X, 'Y':Y, 'R':R, 'S':S }
	all_labels = [0, 1]
	return ClassificationDataset(all_labels, n_candidate, n_safety, n_test, seed=seed, meta_information=meta_information, **contents)

