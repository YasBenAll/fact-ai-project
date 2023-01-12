from itertools import combinations
import numpy as np

def _disparate_impact(c, v1, v2, e, t=1e-20):
	ts = f'%.{np.ceil(max(0,-np.log10(t))).astype(int)}f' % t
	t1 = f'nanmax({ts},E[Yp=1|{c}={v1}])'
	t2 = f'nanmax({ts},E[Yp=1|{c}={v2}])'
	print(t1, t2)
	return f'-min({t1}/{t2}, {t2}/{t1}) - {e}'

CONSTRAINT_BUILDERS = {
	'DemographicParity'     : (lambda c, v1, v2, e: '|E[Yp=1|%s=%d]-E[Yp=1|%s=%d]| - %f' % (c,v1,c,v2,e)),
	# 'DisparateImpact'       : (lambda c, v1, v2, e: '-min(E[Yp=1|%s=%d]/E[Yp=1|%s=%d], E[Yp=1|%s=%d]/E[Yp=1|%s=%d]) - %f' % (c,v1,c,v2,c,v2,c,v1,e)),
	'DisparateImpact' : _disparate_impact,
	'EqualizedOdds'         : (lambda c, v1, v2, e: '|E[Yp=1|Y=1,%s=%d]-E[Yp=1|Y=1,%s=%d]| + |E[Yp=1|Y=-1,%s=%d]-E[Yp=1|Y=-1,%s=%d]| - %f' % (c,v1,c,v2,c,v1,c,v2,e)),
	'EqualOpportunity'      : (lambda c, v1, v2, e: '|E[Yp=-1|Y=1,%s=%d]-E[Yp=-1|Y=1,%s=%d]| - %f' % (c,v1,c,v2,e)),
	'PredictiveEquality'    : (lambda c, v1, v2, e: '|E[Yp=1|Y=-1,%s=%d]-E[Yp=1|Y=-1,%s=%d]| - %f' % (c,v1,c,v2,e)),
}

def make_constraints(definition, name, values, epsilon):
	constraints = []
	f = CONSTRAINT_BUILDERS[definition]
	return [ f(name,v1,v2,epsilon) for (v1,v2) in combinations(values,2) ]

def list_constraint_names():
	return list(CONSTRAINT_BUILDERS.keys())