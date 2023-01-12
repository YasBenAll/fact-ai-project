from .rvs import ConstraintManager

def get_classification_cm(constraints, importance_samplers=None):
	keywords = {
			'FPR' : 'E[Yp=1|Y=-1]',
			'FNR' : 'E[Yp=-1|Y=1]',
			'TPR' : 'E[Yp=1|Y=1]',
			'TNR' : 'E[Yp=-1|Y=-1]',
			'PR'  : 'E[Yp=1]',
			'NR'  : 'E[Yp=-1]'
		}
	constraint_strs = [constraints] if isinstance(constraints, str) else constraints
	return ConstraintManager(constraint_strs, keywords=keywords, importance_samplers=importance_samplers)