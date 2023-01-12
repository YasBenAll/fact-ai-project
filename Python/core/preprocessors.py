import numpy as np

class ClassificationPreprocessor():
	def __init__(self):
		self._defined_vars = ['ER','PR','NR','FPR','FNR','TPR','TNR']

	@property
	def defined_vars(self):
		return self._defined_vars

	@property
	def trivial_bounds(self):
		return { n:(0,1) for n in self.defined_vars }
	
	def process(self, reqd_rvs, X, Y, Yp):
		out = {}
		for name in reqd_rvs:
			# Error rate 
			if name == 'ER':
				out[name] = 1.0 * (Yp != Y)
			# Overall positive rate
			elif name == 'PR':
				out[name] = 1.0 * (Yp == 1)
			# Overall negative rate
			elif name == 'NR':
				out[name] = 1.0 * (Yp != 1)
			# Overall false positive rate
			elif name == 'FPR':
				out[name] = 1.0 * np.logical_and(Yp==1, Y!=1)
			# Overall true positive rate
			elif name == 'FNR':
				out[name] = 1.0 * np.logical_and(Yp!=1, Y==1)
			# Overall false negative rate
			elif name == 'TPR':
				out[name] = 1.0 * np.logical_and(Yp==1, Y==1)
			# Overall true negative rate
			elif name == 'TNR':
				out[name] = 1.0 * np.logical_and(Yp!=1, Y!=1)
		return out


class TypedClassificationPreprocessor():
	def __init__(self, n_types):
		self.n_types = n_types
		types = range(n_types)
		self._defined_vars = [
			'ER' , *[ 'ER_T%d'%t for t in types],
			'PR' , *[ 'PR_T%d'%t for t in types],
			'NR' , *[ 'NR_T%d'%t for t in types],
			'FPR', *['FPR_T%d'%t for t in types],
			'FNR', *['FNR_T%d'%t for t in types],
			'TPR', *['TPR_T%d'%t for t in types],
			'TNR', *['TNR_T%d'%t for t in types] ]
	
	@property
	def defined_vars(self):
		return self._defined_vars

	@property
	def trivial_bounds(self):
		return { n:(0,1) for n in self.defined_vars }

	def process(self, reqd_rvs, X, Y, Yp, T):
		out = {}
		for name in reqd_rvs:
			# Error rate 
			if name == 'ER':
				out[name] = 1.0 * (Yp != Y)

			# Error rate by type
			elif name.startswith('ER_T'):
				tid  = int(name.split('ER_T')[1])
				inds = np.where(T==tid)[0]
				YpT  = Yp[inds].astype(int)
				YT   = T[inds].astype(int)
				out[name] = 1.0 * (YpT != YT)

			# Overall positive rate
			elif name == 'PR':
				out[name] = 1.0 * (Yp == 1)
			# Overall negative rate
			elif name == 'NR':
				out[name] = 1.0 * (Yp != 1)

			# Positive rate per type
			elif name.startswith('PR_T'):
				tid = int(name.split('PR_T')[1])
				YpT = Yp[np.where(T==tid)[0]].astype(int)
				out[name] = 1.0 * (YpT == 1)
			# Negative rate per type
			elif name.startswith('NR_T'):
				tid = int(name.split('NR_T')[1])
				YpT = Yp[np.where(T==tid)[0]].astype(int)
				out[name] = 1.0 * (YpT != 1)

			# Overall false positive rate
			elif name == 'FPR':
				out[name] = 1.0 * np.logical_and(Yp==1, Y!=1)
			# Overall true positive rate
			elif name == 'FNR':
				out[name] = 1.0 * np.logical_and(Yp!=1, Y==1)
			# Overall false negative rate
			elif name == 'TPR':
				out[name] = 1.0 * np.logical_and(Yp==1, Y==1)
			# Overall true negative rate
			elif name == 'TNR':
				out[name] = 1.0 * np.logical_and(Yp!=1, Y!=1)

			# False positive rate per type
			elif name.startswith('FPR_T'):
				tid = int(name.split('FPR_T')[1])
				YpT = Yp[np.logical_and(T==tid, Y!=1)].astype(int)
				out[name] = 1.0 * (YpT == 1)
			# False negative rate per type
			elif name.startswith('FNR_T'):
				tid = int(name.split('FNR_T')[1])
				YpT = Yp[np.logical_and(T==tid, Y==1)].astype(int)
				out[name] = 1.0 * (YpT != 1)
			# True positive rate per type
			elif name.startswith('TPR_T'):
				tid = int(name.split('TPR_T')[1])
				YpT = Yp[np.logical_and(T==tid, Y==1)].astype(int)
				out[name] = 1.0 * (YpT == 1)
			# True negative rate per type
			elif name.startswith('TNR_T'):
				tid = int(name.split('TNR_T')[1])
				YpT = Yp[np.logical_and(T==tid, Y!=1)].astype(int)
				out[name] = 1.0 * (YpT != 1)
		return out
