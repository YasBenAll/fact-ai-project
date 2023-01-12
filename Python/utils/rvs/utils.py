import sys
import code
import time
import joblib
import itertools
import os
import threading
import copy
import numpy as np
import datetime
from scipy.spatial          import ConvexHull
from matplotlib.collections import LineCollection
from matplotlib.colors      import to_rgba_array
from scipy.stats   import norm, chi, t
import re

from scipy import optimize

def check_bounds_vector(bounds):
	contains_intervals = False
	contains_values = False
	for bound in bounds:
		try:
			n_terms = len(bound)	
			assert not(contains_values), 'Bounds vector cannot contain mixtures of intervals and values.'
			assert (n_terms == 2), 'Intervals must contain two values.'
			assert not(np.isclose(bound[0],bound[1])), 'Interval endpoints cannot be identical.'
			assert (bound[1] > bound[0]), 'Interval endpoints must be of the form (lb,ub) with lb < ub.'
			contains_intervals = True
		except TypeError as e:
			assert not(contains_intervals), 'Bounds vector cannot contain mixtures of intervals and values.'
			contains_values = True
	if contains_values:
		assert np.isclose(np.sum(bounds), 1.0), 'Non-interval bounds must sum to 1.0.'
		assert all([ bound >= 0.0 for bound in bounds ]), 'Non-interval bounds must be nonnegative.'
	return contains_intervals

def is_ratio(q, d, known_terms):
	Pr_D_g_C = known_terms['P(D|C)']
	Pr_C_g_D = known_terms['P(C|D)']
	ratio = (Pr_C_g_D[d] * q[d]) / (Pr_C_g_D.dot(q) * Pr_D_g_C[d])
	return 0.0 if np.isnan(ratio) else ratio

def is_estimate(S, D, q, known_terms=None):
	phi_d = np.array([ is_ratio(q,d,known_terms) for d in known_terms['D'] ])
	return np.mean(phi_d[D] * S)

def get_opt_is_estimate_function(S, D, known_terms=None):
	n = len(S)
	vals = np.array([ np.sum(S[D==d])/n for d in known_terms['D'] ])

	Pr_D_g_C = known_terms['P(D|C)']
	Pr_C_g_D = known_terms['P(C|D)']
	is_factor_base = Pr_C_g_D / Pr_D_g_C
	def optf(q):
		is_den = Pr_C_g_D.dot(q)
		phi_d = is_factor_base * q / is_den
		return vals.dot(phi_d)
	return optf

def is_hoeffding(S, D, delta, q, a=0, b=1, known_terms=None, n_scale=1.0):
	phi_d = np.array([ is_ratio(q,d,known_terms) for d in known_terms['D'] ])
	a_q = np.min(phi_d) * a
	b_q = np.max(phi_d) * b
	Sw  = phi_d[D] * S

	nc = np.ceil(len(Sw) * n_scale).astype(int)
	mn = np.mean(Sw)
	offset = (b_q-a_q)* np.sqrt(np.log(2/delta)/nc)
	return safesum(mn,-offset), safesum(mn,offset)
	
def get_opt_is_hoeffding_function(S, D, delta, a=0, b=1, known_terms=None, n_scale=1.0):
	n  = len(S)
	nc = np.ceil(n * n_scale).astype(int)
	vals = np.array([ np.sum(S[D==d])/n for d in known_terms['D'] ])
	offset_scaling_factor = np.sqrt(np.log(2/delta)/nc)

	Pr_D_g_C = known_terms['P(D|C)']
	Pr_C_g_D = known_terms['P(C|D)']
	is_factor_base = Pr_C_g_D / Pr_D_g_C
	def optf(q):
		is_den = Pr_C_g_D.dot(q)
		phi_d = is_factor_base * q / is_den
		a_q = np.min(phi_d) * a
		b_q = np.max(phi_d) * b
		mn  = vals.dot(phi_d)
		offset = (b_q-a_q) * offset_scaling_factor
		return safesum(mn,-offset), safesum(mn,offset)
	return optf

def is_ttest(S, D, delta, q, known_terms, n_scale=1.0):
	phi_d = np.array([ is_ratio(q,d,known_terms) for d in known_terms['D'] ])
	Sw = phi_d[D] * S
	
	nc = np.ceil(len(Sw) * n_scale).astype(int)
	mn  = np.mean(Sw)
	std = np.std(Sw, ddof=1)
	# print('%.4f %.4f %d' % (mn, std, nc))
	offset = std * t.ppf(1-delta/2,nc-1) / np.sqrt(nc-1)
	return safesum(mn,-offset), safesum(mn,offset)

def get_opt_is_ttest_function(S, D, delta, known_terms, n_scale=1.0):
	n  = len(S)
	nc = np.ceil(n * n_scale).astype(int)
	vals = np.array([ np.sum(S[D==d]) for d in known_terms['D'] ])
	
	offset_scaling_factor = t.ppf(1-delta/2,nc-1) / np.sqrt(nc-1)

	Pr_D_g_C = known_terms['P(D|C)']
	Pr_C_g_D = known_terms['P(C|D)']
	is_factor_base = Pr_C_g_D / Pr_D_g_C

	unique_s = np.unique(S)
	counts_s_d = np.array([[ np.sum(np.logical_and(S==s,D==d)) for s in unique_s ] for d in known_terms['D']])
	
	def optf(q):
		is_den = Pr_C_g_D.dot(q)
		phi_d = is_factor_base * q / is_den
		mn  = vals.dot(phi_d) / n
		std = np.sqrt(np.sum((np.outer(phi_d, unique_s) - mn)**2 * counts_s_d)/(n-1))
		offset = std * offset_scaling_factor
		return safesum(mn,-offset), safesum(mn,offset)
	return optf


def get_crossing_edges(C, E, V=None):
	edges = []
	if not(V is None):
		E = E + np.eye(len(E))
		V1, V2 = np.where((V[:,None] == -V[None,:]) * E)
		for (v1,v2) in zip(V1,V2):
			if not((v1,v2) in edges) and not((v2,v1) in edges):
				edges.append((v1,v2))
	else:
		V1, V2 = np.where(E)
		for (v1,v2) in zip(V1,V2):
			if not((v1,v2) in edges) and not((v2,v1) in edges):
				edges.append((v1,v2))
	return edges

def get_simplex_intersection_vertices(C, E):
	C = np.array(C)
	d = len(C[0])
	c = np.sqrt(1/d)
	p = np.ones(d)/np.sqrt(d)

	values = C.dot(p) - c
	V = np.array(np.sign(values), dtype=int)
	V[np.isclose(values,0)] = 0
	edges = get_crossing_edges(C, E, V)

	verts = []
	for e in edges:
		if e[0] != e[1]:
			v1, v2 = C[e[0]], C[e[1]]
			if all(np.isclose(v1,v2)):
				v = v1
			else:
				v = v1 + (v2-v1)*(1-v1.sum())/((v2-v1).sum())
		else:
			v = C[e[0]]
		verts.append(v)
	return np.array(verts)

def get_marginal_region_vertices(B):
	d = len(B)
	V = np.array(list(itertools.product(*B)))
	_A = np.array(list(itertools.product(*[[0,1] for _ in range(d)])))
	E = np.array(np.sum(_A[:,None,:] == _A[None,:,:], axis=-1) == (d-1), dtype=int)
	return get_simplex_intersection_vertices(V,E)
	
def optimize_demographic_shift(values, P_CgD, P_DgC, B, verts=None, return_verts=False):
	# computes min/max of sum_d val[d] Q(D=d|C) / P(D=d|C)
	# where Q(D=d|C) = P(C|D=d) Q(D=d) / ( sum_d' P(C|D=d') Q(D=d') )
	# over all Q(D=d) in (B[d,0], B[d,1])
	vmin, vmax = None, None
	if sum(np.isclose(P_CgD,0)) == len(P_CgD)-1:
		value = values[~np.isclose(P_CgD,0)][0]
		vmin, vmax = value, value
	else:
		if verts is None:
			verts = get_marginal_region_vertices(B)
		Q_DgC = (verts * P_CgD[None,:]) / verts.dot(P_CgD)[:,None]
		vals = (Q_DgC / P_DgC[None,:]).dot(values)
		vmin, vmax = vals.min(), vals.max()
	return (vmin, vmax, verts) if return_verts else (vmin, vmax)


def optimize_on_simplex(f, bounds):
	# print('Using new optimization strategy:')
	# print('   Bounds:')
	# print(bounds)
	# f takes R^d -> R, and is defined on all x \in R^d s.t.:
	#   sum(x) = 1
	#   x_i \in bounds_i for i = 1...d
	# if len(bounds) == 2:
	# 	def _f(_x):
	# 		print('x_in:', _x)
	# 		x = np.array((_x, 1-_x))
	# 		print('x_out:', x)
	# 		print('consts:', sum(x), [l<=xi<=u for xi,(l,u) in zip(x,bounds)])
	# 		sys.stdout.flush()
	# 		return f(x)
	# else:
	def _f(_x):
		x = np.concatenate((_x, [1-_x.sum()]))
		return f(x)

	def callback(_x):
		print('x_in:', _x)
		x = np.concatenate((_x, [1-_x.sum()]))
		print('x_out:', x)
		print('consts:', sum(x), [l<=xi<=u for xi,(l,u) in zip(x,bounds)])
		sys.stdout.flush()

	# _f takes R^(d-1) -> R, and is defined on all x' \in R^(d-1) s.t.:
	#   sum(x') < 1
	#   x'_i \in bounds_i for i = 1...(d-1)
	#   sum(x') \in bounds_d
	_bounds  = bounds[:-1]
	_bl, _bu = bounds[-1]
	_consts =   [ {'type': 'ineq', 'fun': lambda W: 1 - W.sum()},
				  {'type': 'ineq', 'fun': lambda W: 1 - W.sum() - _bl },
				  {'type': 'ineq', 'fun': lambda W: _bu - 1 + W.sum() } ]
	result = optimize.shgo(_f, _bounds, constraints=_consts, callback=None, minimizer_kwargs={'method':'SLSQP', 'constraints':_consts}, options={'disp':False}, sampling_method='sobol')
	# print('Success?:', result.success)
	# print('Status:', result.status)
	# print('Message:', result.message)
	# print('Niters:', result.nit)
	# print('Nevals:', result.nfev)
	# callback(result.x)

	# def g(_x):
	# 	x = np.array([ _x, 1-_x ])
	# 	C = [ (c['fun'](np.array(_x)) >= 0) for c in _consts ]
	# 	D = [ (l<=xi<=u) for xi,(l,u) in zip(x,bounds) ]
	# 	if all(C) and all(D):
	# 		return f(x)
	# 	return np.nan
	# xs = np.linspace(bounds[0][0], bounds[0][1], 100)
	# gs = np.array([ g(_x) for _x in xs ])
	# keyboard()
	return np.concatenate((result.x, [1-result.x.sum()]))

def load_probabilities(B,V,ascending=True):
    I = np.argsort(V)
    if not(ascending):
    	I = I[::-1]
    p = np.zeros(len(B))
    B = np.array(B)[I]
    free = 1.0
    for i, (pmin, pmax) in enumerate(B):
        p[i] = max(pmin, min(pmax,free-B[i+1:,0].sum()))
        free -= p[i]
    p[I] = p.copy()
    return p

def strip_is_funcs(s):
	return re.sub(r'\{[^\}]+\}', '', s)

def replace_keywords(s, repls, include_replacements=False): 
	replacements = {}
	for name, repl in repls.items(): 
		if re.match(r'^E\[.*\]$', repl):
			terms = re.findall(r'(?<![a-zA-Z_])(%s(\{[^\}]+\}|)(\([^\)]+\)|))' % name, s)
			for instance, weightf, condition in terms:
				condition = condition[1:-1] if len(condition) > 0 else ''
				expr   = repl.split('|')[0][2:]
				header = ('E%s['%weightf) if len(weightf) > 0 else 'E['
				new_repl = '%s%s%s' % (header, expr, repl.split(expr)[-1])
				if len(condition) > 0:
					if '|' in repl:
						new_repl = '%s|%s,%s' % (new_repl.split('|')[0], condition, new_repl.split('|')[-1])
					else:
						new_repl = '%s|%s]' % (new_repl[:-1],condition)
				s = s.replace(instance, new_repl, 1)
				replacements[instance] = new_repl
		else:
			s = s.replace(name, repl)

	return (s, replacements) if include_replacements else s

COMPARATORS = {
	'>'  : lambda a, b: a > b,
	'<'  : lambda a, b: a < b,
	'>=' : lambda a, b: a >= b,
	'<=' : lambda a, b: a >= b,
	'='  : lambda a, b: a == b,
	'!=' : lambda a, b: a != b
}

COMPARATOR_NEGATIONS = {
	'>'  : '<=',
	'<'  : '>=',
	'>=' : '<',
	'<=' : '>',
	'='  : '!=',
	'!=' : '='
}

def safesum(a, b):
	a_inf, a_nan = np.isinf(a), np.isnan(a)
	b_inf, b_nan = np.isinf(b), np.isnan(b)
	if (a_nan or b_nan):
		return np.nan
	if a_inf and b_inf and (np.sign(a) != np.sign(b)):
		return np.nan
	return a + b

def safeprod(a, b):
	a_inf, a_nan = np.isinf(a), np.isnan(a)
	b_inf, b_nan = np.isinf(b), np.isnan(b)
	if (a_nan or b_nan):
		return np.nan
	if (a_inf and b==0) or (b_inf and a==0):
		return 0.0
	return a * b

def safediv(a, b):
	a_inf, a_nan = np.isinf(a), np.isnan(a)
	b_inf, b_nan = np.isinf(b), np.isnan(b)
	if (a_nan or b_nan) or (a_inf and b_inf):
		return np.nan
	if (b==0):
		return np.nan
	return a / b

def is_iterable(x):
	try:
		iter(x)
	except Exception:
		return False
	else:
		return True	

def make_seed(digits=8, random_state=np.random):
    return np.floor(random_state.rand()*10**digits).astype(int)

def subdir_incrementer(sd):
	for i in itertools.count():
		yield (sd+'_%d') % i


######################################################
#   Helper for generating reasonable thetas for LC   #
######################################################

def get_linear_samplef(X, buffer_angle=5.0, has_intercept=False, use_chull=True):
	nf = X.shape[1]
	c  = 360.0 / (2*np.pi)
	Z  = X[:,:-1] if has_intercept else X.copy()
	W  = X if not(use_chull) else X[ConvexHull(Z, qhull_options='QJ').vertices]
	WN = W / np.linalg.norm(W,axis=1)[:,None]
	
	def samplef():
		while True:
			t = np.random.normal(0,1,(nf,1))
			t = t / np.linalg.norm(t)
			s = W.dot(t).flatten()
			y = np.sign(s)
			# If the solution shatters the samples, return
			if len(np.unique(y)) == 2:
				return t.flatten()
			# If the DB is close enough to the convex hull, return
			s = WN.dot(t).flatten()
			a = 90.0 - c*np.arccos(np.abs(s).min())
			if a < buffer_angle:
				return t.flatten()
	return samplef
		

#####################################################
#   Helpers for dividing parameters among workers   #
#####################################################

def split_evenly(v, n):
	return [ len(a) for a in np.array_split(np.arange(v),n) ]

def _stack_dicts(base, next, n, replace=False, max_depth=np.inf, na_val=None):
	if isinstance(next,dict) and isinstance(base,dict):
		if max_depth <= 0:
			return np.array([base,next]) if (n>0) else np.array([next])
		out  = {}
		keys = set(base.keys()).union(next.keys())
		for k in keys:
			_base = base[k] if (k in base.keys()) else None
			_next = next[k] if (k in next.keys()) else None
			out[k] = _stack_dicts(_base, _next, n, replace, max_depth-1, na_val=na_val)
	elif isinstance(next,dict) and (base is None):
		out = _stack_dicts({}, next, n, replace, max_depth, na_val=na_val)
	elif isinstance(base,dict) and (next is None):
		out = _stack_dicts(base, {}, n, replace, max_depth, na_val=na_val)
	else:
		if replace:
			out = next if (base is None) else base
		else:
			base_val = np.repeat(na_val,n) if (base is None) else base
			next_val = na_val              if (next is None) else next
			out = np.array(base_val.tolist() + [next_val])
	return out

def stack_all_dicts(*dicts, na_val=None):
	out = {}
	for i,d in enumerate(dicts):
		out = _stack_dicts(out, d, i, max_depth=np.inf, na_val=na_val)
	return out

def stack_all_dicts_shallow(*dicts, na_val=None):
	out = {}
	for i,d in enumerate(dicts):
		out = _stack_dicts(out, d, i, max_depth=1, na_val=na_val)
	return out


#################
#   Profiling   #
#################

class TimeAccumulator(object):
	def __init__(self, n_events=1):
		self.t_last = time.time()
		self.times  = np.zeros(n_events)

	def tick(self, i):
		t = time.time()
		self.times[i] += t - self.t_last
		self.t_last = t

	def total(self, i=None):
		if i is None:
			return sum(self.times)
		return self.times[i]

	def percentages(self):
		tot = self.total()
		if tot == 0.0:
			return np.zeros_like(self.times)
		return 100.0 * self.times/tot

class DurationEstimator(object):
	def __init__(self, min_completed=0.3):
		self._min_completed = min_completed
		self.reset()
	def reset(self):
		self._t = datetime.datetime.utcnow()
	def time_remaining(self, r_completed):
		t_delta = datetime.datetime.utcnow() - self._t
		return t_delta * (1-r_completed) / r_completed
	def time_remaining_str(self, r_completed):
		if r_completed < self._min_completed:
			return 'N/A'
		remainder = self.time_remaining(r_completed).total_seconds()
		weeks, remainder = divmod(remainder, 604800)
		days,  remainder = divmod(remainder, 86400)
		hours, remainder = divmod(remainder, 3600)
		minutes, seconds = divmod(remainder, 60)
		remain_str  = (' %dw' % weeks  ) if weeks   > 0 else ''
		remain_str += (' %dd' % days   ) if days    > 0 else ''
		remain_str += (' %dh' % hours  ) if hours   > 0 else ''
		remain_str += (' %dm' % minutes) if minutes > 0 else ''
		remain_str += (' %ds' % seconds) if seconds > 0 else ''
		return remain_str.strip()


#################
#   Debugging   #
#################

def keyboard(quit=False, banner=''):
	''' Interrupt program flow and start an interactive session in the current frame.
		 * quit   : If True, exit the program upon terminating the session. '''
	try:
		raise None
	except:
		frame = sys.exc_info()[2].tb_frame.f_back
	namespace = frame.f_globals.copy()
	namespace.update(frame.f_locals)
	from sys import exit as quit
	namespace.update({'quit':quit})
	code.interact(banner=banner, local=namespace)
	if quit:
		sys.exit()


#######################################################
#   Plots curves as traces with fading color trails   #
#######################################################

def _get_colors(colors, n):
	colors = to_rgba_array(colors)
	if colors.shape[0] == 1:
		colors = np.tile(colors, (n,1))
	else:
		assert colors.shape[0] == n, 'utils._get_colors: Incorrect number of colors. Expected %d, recieved %d.' % (n,colors.shape[0])
	return colors

def plot_traces(ax, traces, colors='b', linewidths=2):
	# base_colors = plt.cm.jet(pp.minmax_scale(Y)) 
	n_traces    = len(traces)
	base_colors = _get_colors(colors, n_traces)
	lines  = []
	colors = []
	for i, trace in enumerate(traces):
		n_segments = len(trace) - 1
		for j in range(n_segments):
			lines.append(np.array([ trace[j], trace[j+1] ]))
			color = base_colors[i]
			color[-1] = float(j+1) / n_segments
			colors.append(tuple(color))
	lc = LineCollection(lines, colors=colors, linewidths=linewidths)
	ax.add_collection(lc)
	return ax