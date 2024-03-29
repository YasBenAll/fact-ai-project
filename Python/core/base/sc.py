import numpy as np
from functools import partial

from core.optimizers import OPTIMIZERS
from utils.rvs import ConstraintManager # this would be useful to understand

from sys import float_info
MAX_VALUE = float_info.max

from time import time
from utils import keyboard, TimerCollection

class SeldonianClassifierBase:
	def __init__(self, constraint_strs, shape_error, model_type, model_params, verbose=False, ci_mode='hoeffding', robustness_bounds={}, term_values={}, cs_scale=2.0, importance_samplers={}, demographic_variable=None, demographic_variable_values=[], demographic_marginals=[], known_demographic_terms=None, seed=None, robust_loss=False):
		"""
		Parameters
		----------
		constraint_strs : list of str
			List of constraints to be satisfied by the classifier.
		shape_error : float
			Maximum error allowed in the shape of the classifier.
		model_type : str
			Type of model to use. Currently supported: 'linear', 'logistic', 'svm', 'tree', 'forest', 'nn'.
		model_params : dict
			Parameters for the model.
		verbose : bool
			Whether to print out information about the optimization process.
		ci_mode : str
			How to compute confidence intervals. Currently supported: 'hoeffding', 'bootstrap'.
		robustness_bounds : dict
			Bounds on the robustness of the classifier.
		term_values : dict
			Values of the terms in the constraints.
		cs_scale : float
			Scale of the confidence set.
		importance_samplers : dict
			Importance samplers for the constraints.
		demographic_variable : str
			Demographic variable to use.
		demographic_variable_values : list of str
			Values of the demographic variable.
		demographic_marginals : list of float
			Marginals of the demographic variable.
		known_demographic_terms : dict
			Values of the terms in the constraints for each value of the demographic variable.
		seed : int
			Random seed.
		robust_loss : bool
			Whether to use the robust loss function.
		"""
		self.shape_error = shape_error
		self.verbose     = verbose
		self.model_type      = model_type
		self.model_variables = {}
		self.model_params    = model_params
		self.ci_mode = ci_mode
		self._robustness_bounds = robustness_bounds
		self._term_values = term_values
		self._cs_scale = cs_scale
		self._seed = seed
		# Set up the constraint manager to handle the input constraints 
		# keywords is the dictionary of contraint definition, given in line 375 (so this does not work as a standalone class)
		self._cm       = ConstraintManager(constraint_strs, trivial_bounds={}, keywords=self.keywords, importance_samplers=importance_samplers, demographic_variable=demographic_variable, demographic_variable_values=demographic_variable_values, demographic_marginals=demographic_marginals, known_demographic_terms=known_demographic_terms, debug=False, timername='Constraints')
		self._robust_loss = False if demographic_variable is None else robust_loss
		if self._robust_loss:
			self._error_cm = ConstraintManager(['E[Y!=Yp]']   , trivial_bounds={}, keywords=self.keywords, importance_samplers=importance_samplers, demographic_variable=demographic_variable, demographic_variable_values=demographic_variable_values, demographic_marginals=demographic_marginals, known_demographic_terms=known_demographic_terms, debug=False, timername='Loss')

		# Used for profiling
		self._tc = TimerCollection('SeldonianClassifier')


	@property
	def n_constraints(self):
		return self._cm.n_constraints
	
	def load_split(self, split, **extra):
		all_data = { **split, **extra }
		return { k:all_data[k] for k in self._cm.referenced_variables}

	def load_error_split(self, split, **extra):
		all_data = { **split, **extra }
		return { k:all_data[k] for k in self._error_cm.referenced_variables}

	def copy_cm(self, constraint_strs):
		return ConstraintManager(constraint_strs, trivial_bounds=self._cm.trivial_bounds, keywords=self.keywords, importance_samplers=self._cm._importance_samplers, demographic_variable=self._cm._demographic_variable, demographic_variable_values=self._cm._demographic_variable_values, demographic_marginals=self._cm._demographic_marginals, known_demographic_terms=self._cm._known_demographic_terms)

	def set_cm_data(self, predictf, split):
		try:
			Yp = predictf(split['X'])
		except TypeError as e:
			return np.array([np.inf])
		self._cm.set_data(self.load_split(split, Yp=Yp))
		if self._robust_loss:
			self._error_cm.set_data(self.load_error_split(split, Yp=Yp))

	# Error functions

	def _error(self, predictf, X, Y):
		return np.mean(Y != predictf(X))

	def _loss(self, X, Y, theta=None):
		predictf = self.get_predictf(theta=theta)
		return self._error(predictf, X, Y)


	# Accessor for different optimizers

	def get_optimizer(self, name, dataset, opt_params={}):
		if name == 'linear-shatter':
			assert self.model_type == 'linear', 'SeldonianClassifierBase.get_optimizer(): linear-shatter optimizer is only compatible with a linear model.'
			return OPTIMIZERS[name](dataset.X, buffer_angle=5.0, has_intercept=False, use_chull=True)
		elif name == 'cmaes':
			return OPTIMIZERS[name](self.n_features, sigma0=0.01, n_restarts=50, seed=self._seed)
		elif name == 'bfgs':
			return OPTIMIZERS[name](self.n_features, sigma0=2.0, n_restarts=50, **opt_params)
		elif name == 'slsqp':
			return OPTIMIZERS[name](self.n_features)
		raise ValueError('SeldonianClassifierBase.get_optimizer(): Unknown optimizer \'%s\'.' % name)

	
	# Base models	

	def get_predictf(self, theta=None):
		return partial(self.predict, theta=theta)

	def predict(self, X, theta=None):
		'''
		makes model predictions, either with a linear model, a rbf, or a mlp

		interestingly this uses, depending on the model type, also 
		self.model_params['hidden_layers'] or self.model_variables['X']

		Arguments
		---------
		X: np.array
			the predictive variables
		theta: Union[np.array, Tuple]
			model parameters

		Returns
		-------
		np.array 
			The predictions
		'''
		theta = self.theta if (theta is None) else theta
		if self.model_type == 'linear':
			return np.sign(X.dot(theta))
		elif self.model_type == 'rbf':
			s, b, *c = theta
			X_ref = self.model_variables['X']
			Y_ref = self.model_variables['Y']
			P = np.sum((X_ref[:,None,:]-X[None,:,:])**2, axis=2)
			K = np.exp(-0.5*P/(s**2))
			return np.sign(np.einsum('n,n,nm->m', c, Y_ref, K)+b) # not sure what this does but why would they use Y_ref to make predictions?
		elif self.model_type == 'mlp':
			A = X
			for nh in self.model_params['hidden_layers']:
				nw = A.shape[0] * nh
				w, theta = theta[:nw], theta[nw:]
				W = w.reshape((A.shape[0], nh))
				A = np.hstack((A.dot(W),np.ones((W.shape[1],1))))
				A = np.tanh(A)
			return np.sign(A.dot(theta))
		raise ValueError('SeldonianClassifierBase.predict(): unknown model type: \'%s\''%self.model_type)

	@property
	def n_features(self):
		if self.model_type == 'linear':
			return self.dataset.n_features
		if self.model_type == 'rbf':
			return self.dataset.n_optimization + 2
		if self.model_type == 'mlp':
			n = 0
			n0 = self.dataset.n_features-1
			for nh in self.model_params['hidden_layers']:
				n += (n0+1) * nh
				n0 = nh
			n += nh+1
			return n

	def _store_model_variables(self, dataset):
		if self.model_type == 'rbf':
			split = dataset.optimization_splits()
			self.model_variables['X'] = split['X']
			self.model_variables['Y'] = split['Y']

	# Constraint evaluation

	def _loss_using_cm(self):
		'''
		Returns
		------
		float
			upper bound (HCUB) 
		'''
		# If demographic shift contains intervals, this will return the worst case error rate
		return self._error_cm.upper_bound_constraints(None, mode='none', robustness_bounds=self._robustness_bounds, term_values=self._term_values)[0]

	def safety_test(self):
		'''
		Returns
		------
		np.array
			upper bound (HCUB) per constraint
		'''
		return self._cm.upper_bound_constraints(self.deltas, mode=self.ci_mode, interval_scaling=1.0, robustness_bounds=self._robustness_bounds, term_values=self._term_values)
		
	def predict_safety_test(self, data_ratio):
		'''
		
		Parameters
		----------
		data_ratio: float
			I believe its the ratio of data to use to have faster computation?

		Returns
		------
		np.array
			upper bound (HCUB) per constraint
		'''
		return self._cm.upper_bound_constraints(self.deltas, mode=self.ci_mode, interval_scaling=self._cs_scale, n_scale=data_ratio, robustness_bounds=self._robustness_bounds, term_values=self._term_values)

	# Candidate selection

	def candidate_objective(self, theta, split, data_ratio):
		'''
		Computes the upper bound on the error rate for the given model parameters

		Arguments
		---------
		theta: np.array
			model parameters
		split: dict
			dataset split
		data_ratio: float
			I believe its the ratio of data to use to have faster computation?

		Returns
		-------
		float
			upper bound (HCUB) on the error rate
		'''
		self._tc.tic('eval_c_obj')
		self._tc.tic('set_cm_data')
		self.set_cm_data(self.get_predictf(theta), split)
		self._tc.toc('set_cm_data')
		self._tc.tic('c_obj_predict_safety_test')
		sc_ubs = self.predict_safety_test(data_ratio)
		self._tc.toc('c_obj_predict_safety_test')

		# if any of the upper bounds is nan,return high value
		if any(np.isnan(sc_ubs)):
			self._tc.toc('eval_c_obj')
			return MAX_VALUE

		# if all upper bounds <= 0:
		#	if _robust_loss: (I believe this means shifty)
		# 		return HCUB of first contraint
		#	else:
		#		return usual loss
		elif (sc_ubs <= 0.0).all():
			self._tc.tic('c_obj_compute_loss')
			val = self._loss_using_cm() if self._robust_loss else self._loss(split['X'], split['Y'], theta=theta)
			self._tc.toc('c_obj_compute_loss')
			self._tc.toc('eval_c_obj')
			return val

		# what is self.shape_error about?
		# return 1 + upper bound
		elif self.shape_error:
			self._tc.tic('c_obj_shape_error')
			sc_ub_max = np.maximum(np.minimum(sc_ubs, MAX_VALUE), 0.0).sum()
			self._tc.toc('c_obj_shape_error')
			self._tc.toc('eval_c_obj')
			return 1.0 + sc_ub_max
		self._tc.toc('eval_c_obj')
		return 1.0

	# Model training

	def fit(self, dataset, n_iters=1000, optimizer_name='linear-shatter', theta0=None, f=None, opt_params={}):
		'''
		I believe this is a crucial part of the algorithm
		It splits the data, performs candidate selection my minimizing loss,
		and tests if the trained model is indeed fair

		It returns a boolean whether the model is fair or not,
		and saves the model parameters as a class variable

		Parameters
		----------
		dataset: ?
			I guess the dataset whatever that is
		n_iters: int
			number of interations given optimizer will perform
		optimizer_name: str
			name of the classification model to use
		theta0: ?
			probably the init weights for the optimizer
		f: ?
			its also never used anywhere??
		opt_params: dict
			additional parameters for the optimizer

		Returns
		-------
		bool
			Whether a fair model was trained or not
		'''
		self.dataset = dataset
		opt = self.get_optimizer(optimizer_name, dataset, opt_params=opt_params)

		# Store any variables that will be required by the model
		self._store_model_variables(dataset)
		
		# Compute number of samples of g(theta) for the candidate and safety sets
		# Note: This assumes that the number of samples used to estimate g(theta)
		#       doesn't depend on theta itself.  
		data_ratio = dataset.n_safety/dataset.n_optimization # TODO this is where the split ratio is calculated - but what is n_safety and n_optimization?

		# Get the optimizer and set the candidate selection function
		split = dataset.optimization_splits()

		c_objective  = partial(self.candidate_objective, split=split, data_ratio=data_ratio)

		# Perform candidate selection using the optimizer
		self._tc.tic('minimize_c_obj')
		self.theta,_ = opt.minimize(c_objective, n_iters, theta0=theta0)
		self._tc.toc('minimize_c_obj')
		
		# print()
		# self._tc.print_total_times()
		# self._tc.print_avg_times()
		# self._tc.print_num_tics()
		# print()

		# Perform the safety check to determine if theta can be used
		predictf = self.get_predictf()
		self.set_cm_data(predictf, dataset.safety_splits())
		st_thresholds = self.safety_test()
		accept = False if any(np.isnan(st_thresholds)) else (st_thresholds <= 0.0).all()
		return accept # returns false if any threshold is nan or > 0
		# return True

	# Model evaluation

	def evaluate(self, dataset, predictf=None, override_is_seldonian=False):
		'''


		Parameters
		----------
		dataset: ?
			the dataset
		predictf: function
			I believe this is a function to make predictions
		overwwide_is_seldonian: bool
			if no predictf is given, is_seldonian will be set to this value

		Returns
		------
		dict
			containt information whether algorithm can be accepted, 
			confidence levels for the contraints, and safety thresholds 
		'''
		ds_ratio = dataset.n_safety / dataset.n_optimization
		meta   = {}
		splits = {'candidate' : dataset.optimization_splits(),
				  'safety'    : dataset.safety_splits(),
				  'train'     : dataset.training_splits(),
				  'test'      : dataset.testing_splits()}

		# We don't assume to know what model predictf uses, so assume that
		#   that any predictf passed in as an argument is Non-Seldonian
		if predictf is None:
			predictf = self.get_predictf()
			meta['is_seldonian'] = True
		else:
			meta['is_seldonian'] = override_is_seldonian

		# Record statistics for each split of the dataset
		for name, split in splits.items():
			try:
				Yp = predictf(split['X'])
			except TypeError as e:
				meta['loss_%s' % name] = np.nan
				for cnum in range(self.n_constraints):
					meta['co_%d_mean' % cnum] = np.nan
			else:
				meta['loss_%s' % name] = self._error(predictf, split['X'], split['Y'])	
				data = self.load_split(split, Yp=Yp)	
				self._cm.set_data(data)
				values = self._cm.evaluate()
				for cnum, value in enumerate(values):
					meta['co_%d_mean' % cnum] = value
		
		# Record SMLA-specific values or add baseline defaults
		if meta['is_seldonian']:
			self.set_cm_data(predictf, splits['safety'])
			stest      = self.safety_test()
			self.set_cm_data(predictf, splits['candidate'])
			pred_stest = self.predict_safety_test(ds_ratio)
			meta['accept']                = False if any(np.isnan(stest))      else (     stest <= 0.0).all()
			meta['predicted_accept']      = False if any(np.isnan(pred_stest)) else (pred_stest <= 0.0).all()
			for i,(st, pst) in enumerate(zip(stest, pred_stest)):
				meta['co_%d_safety_thresh'  % i] = st
				meta['co_%d_psafety_thresh' % i] = pst
		else:
			meta['accept']                = True
			meta['predicted_accept']      = True
			for i in range(self.n_constraints):
				meta['co_%d_safety_thresh'  % i] = np.nan
				meta['co_%d_psafety_thresh' % i] = np.nan
		return meta

########################
#   Base Classifiers   #
########################

class SeldonianClassifier(SeldonianClassifierBase):
	'''
	this class could be considered the shifty algorithm, but can probably also be used for the other algorithms
	'''
	keywords = { # I believe this is a dictionary of contraint definitions
			'FPR' : 'E[Yp=1|Y=-1]',
			'FNR' : 'E[Yp=-1|Y=1]',
			'TPR' : 'E[Yp=1|Y=1]',
			'TNR' : 'E[Yp=-1|Y=-1]',
			'PR'  : 'E[Yp=1]',
			'NR'  : 'E[Yp=-1]'
		}
	def __init__(self, constraint_strs, deltas, shape_error=False, verbose=False, model_type='linear', model_params={}, ci_mode='hoeffding', robustness_bounds=[], term_values={}, cs_scale=2.0, importance_samplers={}, demographic_variable=None, demographic_variable_values=[], demographic_marginals=[], known_demographic_terms=None, seed=None, robust_loss=False):
		"""
		Parameters
		----------
		constraint_strs: list of strings
			list of constraints to be used
		deltas: list of floats
			list of deltas to be used
		shape_error: bool
			whether to use shape error
		verbose: bool
			whether to print out information
		model_type: string
			type of model to use
		model_params: dict
			dictionary of parameters for the model
		ci_mode: string
			how to compute confidence intervals
		robustness_bounds: list of floats
			list of robustness bounds
		term_values: dict
			dictionary of term values
		cs_scale: float
			scale for confidence set
		importance_samplers: dict
			dictionary of importance samplers
		demographic_variable: string
			demographic variable
		demographic_variable_values: list of strings
			list of demographic variable values
		demographic_marginals: list of floats
			list of demographic marginals
		known_demographic_terms: dict
			dictionary of known demographic terms
		seed: int
			seed for random number generator
		robust_loss: bool
			whether to use robust loss
		"""
		self.deltas = deltas
		super().__init__(constraint_strs, shape_error=shape_error, verbose=verbose, model_type=model_type, model_params=model_params, ci_mode=ci_mode, robustness_bounds=robustness_bounds, term_values=term_values, cs_scale=cs_scale, importance_samplers=importance_samplers, demographic_variable=demographic_variable, demographic_variable_values=demographic_variable_values, demographic_marginals=demographic_marginals, known_demographic_terms=known_demographic_terms, seed=seed, robust_loss=robust_loss)

class SeldonianMCClassifier(SeldonianClassifierBase):
	'''
	this seems to me for multiclass classification, but it wasnt talked about in the paper so we can probably ignore this 
	'''
	def __init__(self, epsilons, deltas, shape_error=False, verbose=False, model_type='linear', n_classes=2, model_params={}, loss_weights=None, robustness_bounds={}, term_values={}):
		self.n_classes = n_classes
		self.loss_weights = 1 - np.eye(n_classes) if (loss_weights is None) else loss_weights
		self.constraint_weights = np.zeros_like(self.loss_weights)
		self.constraint_weights[0,1] = 1.0
		self.epsilons = epsilons
		self.deltas   = deltas
		super().__init__(shape_error=shape_error, verbose=verbose, model_type=model_type, model_params=model_params, robustness_bounds=robustness_bounds, term_values=term_values)

	def predict(self, X, theta=None):
		theta = self.theta if (theta is None) else theta
		if self.model_type == 'linear':
			theta = theta.reshape((X.shape[1],self.n_classes))
			return np.argmax(X.dot(theta), axis=1)
		raise ValueError('GeneralSeldonianMCClassifier.predict(): unknown model type: \'%s\''%self.model_type)

	def _get_confusion_indicators(self, Y, Yp):
		n = len(Y)
		C  = np.zeros((n,self.n_classes,self.n_classes))
		C[np.arange(n), Yp, Y] += 1
		return C

	def _error(self, predictf, X, Y):
		Yp = predictf(X)
		C = self._get_confusion_indicators(Y, Yp)
		return (C.sum(0) * self.loss_weights).sum() / C.shape[0]

	@property
	def n_features(self):
		if self.model_type == 'linear':
			return self.dataset.n_features * self.n_classes
			