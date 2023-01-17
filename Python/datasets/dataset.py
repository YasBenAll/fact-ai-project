import numpy as np
import types
from copy import deepcopy
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


class Dataset(object):
	def __init__(self, n_candidate, n_safety, n_test, seed=None, meta_information={}, **contents):
		# Record dataset split sizes
		self._n_safety    = n_safety 
		self._n_candidate = n_candidate
		self._n_test      = n_test
		self._n_train     = n_candidate + n_safety
		self._n_samples   = self._n_train + n_test

		self._seed = seed
		self._meta_information = meta_information

		self._contents = deepcopy(contents)
		self._unique_values = {}
		for k, v in contents.items():
			setattr(self, '_%s' % k, v)

			if v.dtype == np.dtype(np.int32):
				self._unique_values[k] = np.unique(v)

		# Compute indices for the splits
		self._inds = {
			'all'   : np.arange(0, self._n_samples),
			'train' : np.arange(0, self._n_train),
			'test'  : np.arange(self._n_train, self._n_samples),
			'opt'   : np.arange(0, self._n_candidate), ###optimization (candidate) split
			'saf'   : np.arange(self._n_candidate, self._n_train) ###safety split: train - candidate
		}

	@property
	def n_train(self):
		###returns number of training samples
		return len(self._inds['train'])
	@property
	def n_test(self):
		###returns number of test samples
		return len(self._inds['test'])
	@property
	def n_optimization(self):
		###returns number of optimization samples
		return len(self._inds['opt'])
	@property
	def n_safety(self):
		###returns number of safety samples
		return len(self._inds['saf'])

	def _get_splits(self, index_key, keys=None):
		"""
	Get the splits for the data

	Args:
		index_key: index of the key for the split. Available incides are:
		all, train, test, opt, saf
		keys: number of keys for the split

	Returns:
		dict: return dictionary of data according to the split given
	"""
		keys = self._contents.keys() if (keys is None) else keys
		inds = self._inds[index_key]
		return { k:self._contents[k][inds] for k in keys }
	def all_sets(self, keys=None):
		###returns all data
		return self._get_splits('all', keys=keys)
	def training_splits(self, keys=None):
		###returns training data
		return self._get_splits('train', keys=keys)
	def testing_splits(self, keys=None):
		###returns test data
		return self._get_splits('test', keys=keys)
	def optimization_splits(self, keys=None):
		###returns optimization data
		return self._get_splits('opt', keys=keys)
	def safety_splits(self, keys=None):
		###returns safety data
		return self._get_splits('saf', keys=keys)


class ClassificationDataset(Dataset):
	def __init__(self, all_labels, n_candidate, n_safety, n_test, seed=None, meta_information={}, **contents):
		assert 'X' in contents.keys(), 'ClassificationDataset.__init__(): Feature matrix \'X\' is not defined.'
		assert 'Y' in contents.keys(), 'ClassificationDataset.__init__(): Label vector \'Y\' is not defined.'
		super().__init__(n_candidate, n_safety, n_test, seed=seed, meta_information=meta_information, **contents)
		self._labels = np.unique(all_labels)
	@property
	def n_features(self):
		###returns number of data features for the classifier
		return self._X.shape[1]
	@property
	def n_labels(self):
		###returns number of data labels for the classifier
		return len(self._labels)

	def resample(self, n_candidate=None, n_safety=None, n_test=None, probf=None):
		"""
	Resample the classification data. (create new data based on observed data)

	Args:
		n_candidate: number of candidate samples
		n_safety: number of safety samples
		n_test: number of test samples
		probf: probability for the resampling

	Returns:
		ClassificationDataset object: return ClassificationDataset object with the resampled data
	"""
		n_candidate = self._n_candidate if n_candidate is None else n_candidate
		n_safety = self._n_safety if n_safety is None else n_safety
		n_test = self._n_test if n_test is None else n_test
		n = len(self._X)
		rand = np.random.RandomState(self._seed)
		if probf is None:
			P = np.ones(n) / n
		else:
			P = np.array([ probf(i,x,y,t) for i,(x,y,t) in enumerate(zip(self._X, self._R, self._T)) ])

		I = rand.choice(n, n_candidate+n_safety+n_test, replace=True, p=P)
		contents = { k:v[I] for k,v in self._contents.items() }
		output = ClassificationDataset(self._labels, n_candidate, n_safety, n_test, seed=self._seed, meta_information=self._meta_information, **contents)
		output._unique_values = deepcopy(self._unique_values)
		return output


class RLDataset(object):
	def __init__(self, S, A, R, T, P, n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, gamma=1.0, seed=None, Rc_func=(lambda s,a,r,t: r)):
		"""
		This class is used to store the data for reinforcement learning.

		args:
			S: state
			A: action
			R: reward
			T: terminal
			P: probability
			n_actions: number of actions
			n_candidate: number of candidate samples
			n_safety: number of safety samples
			n_test: number of test samples
			min_reward: minimum reward
			max_reward: maximum reward
			gamma: discount factor
			seed: random seed
			Rc_func: reward correction function
		
		output: RLDataset object. This object contains the data for reinforcement learning.
		"""
		n_train   = n_candidate + n_safety
		n_samples = n_train + n_test
		# Store the base datasets
		T = T if not(T is None) else np.zeros(len(S))
		self.gamma = gamma
		self._S = S 
		self._A = A 
		self._R_raw = R 
		self._Rc = np.array([ Rc_func(s,a,r,t) for (s,a,r,t) in zip(S,A,R,T) ]) 
		self._apply_corrections = True
		self._T = T 
		self._P = P 
		self.n_actions = n_actions
		self.max_reward = max_reward
		self.min_reward = min_reward
		# Compute indices for the splits
		self._inds = {
			'all'   : np.arange(0, n_samples),
			'train' : np.arange(0, n_train),
			'test'  : np.arange(n_train, n_samples),
			'opt'   : np.arange(0, n_candidate), 
			'saf'   : np.arange(n_candidate, n_train) 
		}
		# Compute indices for T=0/T=1 splits
		for k, inds in list(self._inds.items()):
			self._inds['%s_0'%k] = inds[T[inds]==0] # T=0
			self._inds['%s_1'%k] = inds[T[inds]==1] # T=1
		# Store the default seed
		self._seed = seed

	def enable_R_corrections(self):
		self._apply_corrections = True 

	def disable_R_corrections(self):
		self._apply_corrections = False

	@property
	def _R(self):
		"""Return the reward vector, possibly corrected."""
		if self._apply_corrections:
			return self._Rc
		return self._R_raw

	@property
	def n_features(self):
		"""Return number of data features"""
		return self._S[0].shape[1]
	@property
	def n_train(self):
		"""Return number of training samples"""

		return len(self._inds['train'])
	@property
	def n_test(self):
		"""Return number of test samples"""
		return len(self._inds['test'])
	@property
	def n_optimization(self):
		"""Return number of optimization samples"""
		return len(self._inds['opt'])
	@property
	def n_safety(self):
		"""Return number of safety samples"""
		return len(self._inds['saf'])

	def _get_splits(self, index_key, t=None, corrected_R=True):
		"""
		Return the splits for the given index key. This function returns the splits
		args:
			index_key: key for the split
			t: time step
			corrected_R: whether to use the corrected reward

		output: S, A, R, T, P. S is the state, A is the action, R is the reward, T is the terminal, P is the probability.
		
		"""
		if not(t is None):
			index_key += ('_%d' % t)
		inds = self._inds[index_key]
		R = self._R[inds] if corrected_R else self._R_raw[inds]
		return self._S[inds], self._A[inds], R, self._T[inds], self._P[inds]

	def _get_splits_by_type(self, index_key, truncate=True, reorder=False, seed=None, corrected_R=True):
		"""
		Return the splits for the given index key. This function returns the splits for T=0 and T=1 separately.
		args:
			index_key: key for the split
			truncate: whether to truncate the splits to the same length
			reorder: whether to reorder the splits
			seed: random seed
			corrected_R: whether to use the corrected reward

		output: S0, A0, R0, T0, P0, S1, A1, R1, T1, P1. S0 is the state for T=0, A0 is the action for T=0, R0 is the reward for T=0, T0 is the terminal for T=0, P0 is the probability for T=0. S1 is the state for T=1, A1 is the action for T=1, R1 is the reward for T=1, T1 is the terminal for T=1, P1 is the probability for T=1.
		
		"""
		S0, A0, R0, _ = self._get_splits(index_key, t=0, corrected_R=corrected_R)
		S1, A1, R1, _ = self._get_splits(index_key, t=1, corrected_R=corrected_R)
		if reorder:
			# Reorder the splits. This is useful for training.
			rnd = np.random.RandomState(self._seed if (seed is None) else seed)
			I0 = rnd.choice(S0.shape[0], S0.shape[0], replace=False)
			I1 = rnd.choice(S1.shape[0], S1.shape[0], replace=False)
			S0, A0, R0, P0 = S0[I0], A0[I0], R0[I0], P0[I0]
			S1, A1, R1, P1 = S1[I1], A1[I1], R1[I1], P1[I1]
		if truncate:
			# Truncate the splits to the same length. This is useful for testing.
			k = min(S0.shape[0], S1.shape[0])
			S0, A0, R0, P0 = S0[:k], A0[:k], R0[:k], P0[:k]
			S1, A1, R1, P1 = S1[:k], A1[:k], R1[:k], P1[:k]
		return S0, A0, R0, P0, S1, A1, R1, P1 # T is not returned because it is always 0 or 1

	@property
	def S(self):
		"""Return the state vector. This is a copy of the original data. """
		return self._S.copy()
	@property 
	def A(self):
		"""Return the action vector. This is a copy of the original data."""
		return self._A.copy()
	@property
	def R(self):
		"""Return the reward vector. This is a copy of the original data."""
		return self._R.copy()
	@property
	def T(self):
		"""Return the terminal vector. This is a copy of the original data."""
		return self._T.copy()
	@property
	def P(self):
		"""Return the probability vector. This is a copy of the original data."""
		return self._P.copy()

	def all_sets(self, t=None, corrected_R=True):
		"""Returns all data"""
		return self._get_splits('all', t=t, corrected_R=corrected_R)
	def training_splits(self, t=None, corrected_R=True):
		"""Returns training data"""
		return self._get_splits('train', t=t, corrected_R=corrected_R)
	def testing_splits(self, t=None, corrected_R=True):
		"""Returns testing data"""
		return self._get_splits('test', t=t, corrected_R=corrected_R)
	def optimization_splits(self, t=None, corrected_R=True):
		"""Returns optimization data. This is the same as training data."""
		return self._get_splits('opt', t=t, corrected_R=corrected_R)
	def safety_splits(self, t=None, corrected_R=True):
		"""Returns safety data. This is the same as testing data."""
		return self._get_splits('saf', t=t, corrected_R=corrected_R)

	def all_sets_by_type(self, truncate=True, reorder=False, seed=None):
		"""Returns all data. This is the same as all_sets()."""
		return self._get_splits_by_type('all', truncate=truncate, reorder=reorder, seed=seed)
	def training_splits_by_type(self, truncate=True, reorder=False, seed=None):
		"""Returns training data. This is the same as training_splits()."""
		return self._get_splits_by_type('train', truncate=truncate, reorder=reorder, seed=seed)
	def testing_splits_by_type(self, truncate=True, reorder=False, seed=None):
		"""Returns testing data. This is the same as testing_splits()."""
		return self._get_splits_by_type('test', truncate=truncate, reorder=reorder, seed=seed)
	def optimization_splits_by_type(self, truncate=True, reorder=False, seed=None):
		"""Returns optimization data. This is the same as optimization_splits()."""
		return self._get_splits_by_type('opt', truncate=truncate, reorder=reorder, seed=seed)
	def safety_splits_by_type(self, truncate=True, reorder=False, seed=None):
		"""Returns safety data. This is the same as safety_splits()."""
		return self._get_splits_by_type('saf', truncate=truncate, reorder=reorder, seed=seed)




class BanditDataset(RLDataset):
	def __init__(self, S, A, R, n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, seed=None, P=None, T=None, Rc_func=(lambda s,a,r,t: r)):
		"""Initialize a dataset for a bandit problem. This is a special case of the RLDataset class. 
		Parameters:
			S: State vector. This is a 2D array of shape (n_samples, n_features).
			A: Action vector. This is a 1D array of shape (n_samples,).
			R: Reward vector. This is a 1D array of shape (n_samples,).
			n_actions: Number of actions.
			n_candidate: Number of candidate actions to use for optimization.
			n_safety: Number of candidate actions to use for safety testing.
			n_test: Number of candidate actions to use for testing.
			min_reward: Minimum reward.
			max_reward: Maximum reward.
			seed: Random seed.
			P: Probability vector. This is a 1D array of shape (n_samples,). If not provided, the reference probabilities are computed using a Gaussian process.
			T: Terminal vector. This is a 1D array of shape (n_samples,). If not provided, it is assumed that all samples are non-terminal.
			Rc_func: Reward correction function. This is a function that takes as input the state, action, reward, and terminal vectors and returns the corrected reward vector.
			
			Example:
				def Rc_func(S,A,R,T):
					return R + 0.1 * np.random.randn(len(R))
			
			output: Corrected reward vector. This is a 1D array of shape (n_samples,). """

		S = S[:,None,:] # Convert S into a sequence of length-1 trajectories
		A = A[:,None]
		# Compute reference probabilities if they aren't provided
		if P is None:
			# Compute reference probabilities using a Gaussian process. This is necessary for consistency with the other datasets.
			self.train_proba_gp()
			X = np.hstack((self._S[:,0,:],self._T[:,None]))
			Ps = self._proba_gp.predict_proba(X)
			P = np.array([ [Ps[i,a]] for i,a in enumerate(A) ])
		else:
			# Convert P into a sequence of length-1 trajectories. This is necessary for consistency with the other datasets.
			P = P[:,None]
			self._proba_gp = None
		self._return_gp = None
		super().__init__(S, A, R, T, P, n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, seed=seed, gamma=1.0, Rc_func=Rc_func) 

	def train_proba_gp(self, use_pct=0.1):
		"""Trains a Gaussian process to predict the reference probabilities.
		Parameters:
			use_pct: Percentage of data to use for training.
		output: None."""
		kernel = 1.0 * RBF(1.0)
		self._proba_gp = GaussianProcessClassifier(kernel) 
		X = np.hstack((self._S[:,0,:],self._T[:,None]))
		I = np.arange(X.shape[0])
		np.random.shuffle(I)
		n_attempts = 0
		n_train = int(use_pct*X.shape[0])
		while len(np.unique(self._A[I[:n_train]])) < self.n_actions and n_attempts < 100:
			np.random.shuffle(I)
			n_attempts += 1
		if len(np.unique(self._A[I[:n_train]])) < self.n_actions and n_attempts == 100:
			raise RuntimeError('Unable to train GP on a representative sample of actions')
		I = I[:n_train]
		self._proba_gp.fit(X[I],self._A[I][:,0])

	def train_return_gp(self, returns, use_pct=0.1):
		"""Trains a Gaussian process to predict the return. 
		Parameters:
			returns: List of return values.
			use_pct: Percentage of data to use for training.
			output: None.
		
		output: None."""
		Y = np.zeros(len(self._R))
		for i,r in enumerate(self._R):
			Y[i] = np.where(r==returns)[0]
		kernel = 1.0 * RBF(1.0)
		self._return_gp = GaussianProcessClassifier(kernel)

		X = np.hstack((self._S[:,0,:],self._T[:,None],self._A))
		n_train = int(use_pct*X.shape[0])
		I = np.arange(X.shape[0])
		np.random.shuffle(I)
		I = I[:n_train]
		self._return_gp.fit(X[I],Y[I]) # Train the GP


	def _get_splits(self, index_key, t=None, flatten=False, corrected_R=True):
		"""Returns the specified split of the dataset. 
		Parameters:
			index_key: Key for the split to return.
			t: Time index. If None, all time indices are returned.
			flatten: If True, the returned arrays are flattened.
			corrected_R: If True, the corrected reward vector is returned.
			output: S, A, R, T, P arrays.
			
		output: S, A, R, T, P arrays."""
		if not(t is None):
			index_key += ('_%d' % t)
		inds = self._inds[index_key]
		S = self._S[inds][:,0,:] if flatten else self._S[inds]
		A = self._A[inds][:,0]   if flatten else self._A[inds]
		P = self._P[inds][:,0]   if flatten else self._P[inds]
		R = self._R[inds] if corrected_R else self._R_raw[inds]
		return S, A, R, self._T[inds], P

	def all_sets(self, t=None, flatten=False, corrected_R=True):
		"""Returns all sets of the dataset."""
		return self._get_splits('all', t=t, flatten=flatten, corrected_R=corrected_R)
	def training_splits(self, t=None, flatten=False, corrected_R=True):
		"""Returns the training set of the dataset."""
		return self._get_splits('train', t=t, flatten=flatten, corrected_R=corrected_R)
	def testing_splits(self, t=None, flatten=False, corrected_R=True):
		"""Returns the testing set of the dataset."""
		return self._get_splits('test', t=t, flatten=flatten, corrected_R=corrected_R)
	def optimization_splits(self, t=None, flatten=False, corrected_R=True):
		"""Returns the optimization set of the dataset."""
		return self._get_splits('opt', t=t, flatten=flatten, corrected_R=corrected_R)
	def safety_splits(self, t=None, flatten=False, corrected_R=True):
		"""Returns the safety set of the dataset."""
		return self._get_splits('saf', t=t, flatten=flatten, corrected_R=corrected_R)

	def get_simulator(self, use_classification=False):
		# Train the predictor for reference probabilities if it hasn't been defined yet
		if self._proba_gp is None:
			self.train_proba_gp()

		# Train the predictor for returns if it hasn't been defined yet
		returns = np.unique(self._R)
		if self._return_gp is None:
			self.train_return_gp(returns)

		def predict(S, A, T):
			"""Predicts the next state, reward, and probability of success.
			Parameters:
				S: Current state.
				A: Current action.
				T: Current time.
					
			output: Next state, action, reward, time, and probability of success."""
			if S.ndim == 1:
				# If the state is a single vector, then we need to reshape it
				X = np.hstack((S[None,:],np.array([[T]])))
				P = self._proba_gp.predict(X).astype(int)
				X = np.hstack((X, np.array([[A]])))
				R = returns[self._return_gp.predict(X).astype(int)]
				return S, A, R[0], T, P[0]
			else:
				# Otherwise, we can just stack the state and time
				X = np.hstack((S[:,0,:],T[:,None]))
				P = self._proba_gp.predict(X).astype(int)[:,None]
				X = np.hstack((X, A))
				R = returns[self._return_gp.predict(X).astype(int)]
				return S, A, R, T, P 
		return predict