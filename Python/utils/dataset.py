import numpy as np
from utils.rvs import ConstraintManager, get_parser


def standardized(X):
	if X.ndim == 1:
		X = X - X.mean(0)
		X = X / X.std(0,ddof=1)
	else:
		X = X - X.mean(0)[None,:]
		X = X / X.std(0,ddof=1)[None,:]
	return X

def with_intercept(X):
	return with_feature(X, np.ones(len(X)))

def with_feature(X, T):
	if X.ndim == 1:
		X = X[:,None]
	if T.ndim == 1:
		T = T[:,None]
	return np.hstack((X, T))