import numpy as np
from scipy import stats
from scipy.optimize import minimize, LinearConstraint
from time import time

import warnings

import sys, code
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

def cCGF(c, V, p):
	''' Computes the centered cumulant generating function. '''
	EV  = np.sum(p * V)
	# A numerically stable computation
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		offset = (c*V).max()
		return (offset-c*EV) + np.log(np.sum([ _p * np.exp(c*v-offset) for (_p,v) in zip(p,V) if _p > 0 ])) 
	# MGF = np.sum([ _p * np.exp(c*(v-EV)) for (_p,v) in zip(p,V) if _p > 0 ])
	# return np.log(MGF)

def grad_cCGF_dP(c, V, p):
	''' Computes the gradien of the cCGF w.r.t. p.'''
	_terms = c*(V-np.sum(p * V))
	if any(_terms > 100):
		_terms = _terms - _terms.max()
	terms = np.exp(_terms)
	MGF   = np.sum(p*terms)
	out =  terms/MGF - c*V
	return out

def goodman_bound(p, n, k, alpha):
	''' Calculates simultaneous bounds on the multinomial proportion (Goodman, 1950?). '''
	cinv = stats.chi2.ppf(1-alpha/k, 1)
	a = cinv + n
	b = -(cinv + 2*n*p)
	c = n*p**2
	root = np.sqrt(b**2 - 4*a*c)
	return (-b-root)/(2*a), (-b+root)/(2*a)

def get_largest_step_component_bounds(p, G, c, V, pmins, pmaxs, max_step=np.inf, debug=False):
	''' Calculates the largest step that can be taken before a constraint is violated. '''
	for (g,_p,pmin,pmax) in zip(G,p,pmins,pmaxs):
		if np.isclose(g,0):
			pass
		elif g > 0:
			max_step = min(max_step,  (pmax-_p)/g)
		else:
			max_step = min(max_step, -(_p-pmin)/g)
	return max_step

def project_grad_component_bounds(G, p, pmins, pmaxs):
	''' Removes components of G that point in directions that violate constraints. '''
	Gp = np.zeros_like(G)
	J_mins = np.isclose(pmins,p)
	J_maxs = np.isclose(pmaxs,p)

	k = len(p)
	E = (np.eye(k) * k - np.ones((k,k))) / np.sqrt(k**2-k)

	c = E.dot(G)
	J = np.logical_or( np.logical_and(np.isclose(pmins,p), c<0), 
					   np.logical_and(np.isclose(pmaxs,p), c>0))
	def f(g, G=G):
		r = G - g
		return r.dot(r), -2*r

	if any(J_mins) and any(J_maxs):
		n_viols = sum(J_mins) + sum(J_maxs)
		lc = LinearConstraint( np.vstack(( E[J_maxs], -E[J_mins] )), -np.inf*np.ones(n_viols), np.zeros(n_viols) )
		Gp = minimize(f, args=(G,), x0=np.zeros_like(G), jac=True, constraints=[lc]).x
	elif any(J_mins):
		n_viols = sum(J_mins)
		lc = LinearConstraint( -E[J_mins], -np.inf*np.ones(n_viols), np.zeros(n_viols) )
		Gp = minimize(f, args=(G,), x0=np.zeros_like(G), jac=True, constraints=[lc]).x
	elif any(J_maxs):
		n_viols = sum(J_maxs)
		lc = LinearConstraint( E[J_maxs], -np.inf*np.ones(n_viols), np.zeros(n_viols) )
		Gp = minimize(f, args=(G,), x0=np.zeros_like(G), jac=True, constraints=[lc]).x
	else:
		Gp = G
	Gp[J_mins] = np.maximum(Gp[J_mins],0)
	Gp[J_maxs] = np.minimum(Gp[J_maxs],0)
	Gp[~np.logical_or(J_mins,J_maxs)] = Gp[~np.logical_or(J_mins,J_maxs)] - Gp.sum() 
	return Gp


def get_largest_step_ev_bounds(p, G, c, V, evmin, evmax, max_step=np.inf, debug=False):
	''' Calculates the largest step that can be taken before a constraint is violated. '''
	if np.isclose(V.dot(G), 0):
		pass
	elif V.dot(G) > 0:
		max_step = min(max_step, (evmax-sum(p*V))/sum(G*V))
	elif V.dot(G) < 0:
		max_step = min(max_step, (evmin-sum(p*V))/sum(G*V))

	for (g,_p) in zip(G,p):
		if np.isclose(g,0):
			pass
		elif g > 0:
			max_step = min(max_step,  (1-_p)/g)
		else:
			max_step = min(max_step, -(_p-0)/g)
	return max_step

def project_grad_ev_bounds(G, p, V, evmin, evmax):
	''' Removes components of G that point in directions that violate constraints. '''
	

	J_mins = np.isclose(0,p)
	J_maxs = np.isclose(1,p)
	k = len(p)

	def f(g, G=G):
		r = G - g
		return r.dot(r), -2*r

	c_As = []
	c_Bs = []
	c_Cs = []
	E = (np.eye(k) * k - np.ones((k,k))) / np.sqrt(k**2-k)
	if any(J_mins):
		c_As.append(E[J_mins])
		c_Bs.append(np.zeros(sum(J_mins)))
		c_Cs.append(np.inf * np.ones(sum(J_mins)))
	if any(J_maxs):
		c_As.append(E[J_maxs])
		c_Bs.append(-np.inf * np.ones(sum(J_maxs)))
		c_Cs.append(np.zeros(sum(J_maxs)))

	EV = sum(p*V)
	d = V - V.mean()
	d = d / np.linalg.norm(d)
	if np.isclose(EV,evmin) or (EV < evmin):
		c_As.append(d[None,:])
		c_Bs.append(np.array([ 0 ]))
		c_Cs.append(np.array([ np.inf ]))
	elif np.isclose(EV,evmax) or (EV > evmax):
		c_As.append(d[None,:])
		c_Bs.append(np.array([ -np.inf ]))
		c_Cs.append(np.array([ 0 ]))

	if len(c_As) == 0:
		Gp = G
	else:
		c_As = np.concatenate(c_As)
		c_Bs = np.concatenate(c_Bs)
		c_Cs = np.concatenate(c_Cs)
		lc = LinearConstraint( c_As, c_Bs, c_Cs )
		Gp = minimize(f, args=(G,), x0=np.zeros_like(G), jac=True, constraints=[lc]).x

	Gp[J_mins] = np.maximum(Gp[J_mins],0)
	Gp[J_maxs] = np.minimum(Gp[J_maxs],0)
	Gp[~np.logical_or(J_mins,J_maxs)] = Gp[~np.logical_or(J_mins,J_maxs)] - Gp.sum() 
	return Gp

def step_over_P(c, e, V, p, projectionf, stepf, upper=True):	
	''' Takes a step to maxmize the upper bound or minimize the lower bound w.r.t. p.'''
	if np.isclose(e,0) and upper:
		def gradf(p):
			return projectionf(V-V.mean(), p)
	elif np.isclose(e,0):
		def gradf(p):
			return projectionf(-V+V.mean(), p)
	else:
		_c = c if upper else -c
		Z = _c*V
		E = np.maximum(np.exp(Z - Z.max()), 1e-20)
		def gradf(p,c=_c,Z=Z,E=E,V=V):
			G = (1/np.abs(c)) * E / p.dot(E)
			G = G - G.mean()
			return projectionf(G, p)

	G = gradf(p)
	if not(all(np.isclose(G,0))):
		step = 0.9*stepf(p, G, c, V, max_step=np.inf)
		p = p + step*G
	return p


def maximize_over_P(c, e, V, p, projectionf, stepf, step_ratio=1.0, upper=True):	
	''' Takes a step to maxmize the upper bound or minimize the lower bound w.r.t. p.'''
	if np.isclose(e,0) and upper:
		def gradf(p):
			return projectionf(V-V.mean(), p)
	elif np.isclose(e,0):
		def gradf(p):
			return projectionf(-V+V.mean(), p)
	else:
		_c = c if upper else -c
		Z = _c*V
		E = np.maximum(np.exp(Z - Z.max()), 1e-20)
		def gradf(p,c=_c,Z=Z,E=E,V=V):
			G = (1/np.abs(c)) * E / p.dot(E)
			G = G - G.mean()
			return projectionf(G, p)
	ps = [p]
	i = 0
	G = gradf(p)
	while not(all(np.isclose(G,0))):
		step = step_ratio*stepf(p, G, c, V, max_step=np.inf)
		p = p + step*G
		ps.append(p)
		G = gradf(p)
	return p

def minimize_over_c(c, e, V, p, upper=True):
	''' Computes the c that defines the upper/lower bound. '''
	if np.isclose(e,0):
		return 0.0

	def invLF(c, e, V, p, upper=upper):
		if np.isclose(c, 0) and not(np.isclose(e,0)):
			return np.inf
		if upper:
			return (cCGF( c,V,p) + e) / c
		else:
			return (cCGF(-c,V,p) + e) / c


	c_start = c
	v_start = invLF(c_start, e, V, p) 	
	alpha = 1e-12
	while not(np.isclose(c_start,0)) and (invLF(c_start+alpha, e, V, p) >= v_start):
		c_start = c_start / 2
		v_start = invLF(c_start, e, V, p) 	

	
	# Perform a line search to find an interval containing the optimum
	c_0 = c_start
	v_0 = v_start
	c_1 = c_start
	v_1 = v_start
	i = 0
	while True:
		c_2 = c_start + alpha
		v_2 = invLF(c_2, e, V, p)
		if c_2 >= 1e100:
			return c_2
		if np.isnan(v_2) or v_2 > v_1: 
			break
		else:
			if i > 0:
				alpha = alpha*10
			c_0 = c_1
			v_0 = v_1
			c_1 = c_2
			v_1 = v_2
			i += 1

	# We now have v_1 < v_0 and v_1 < v_2: finish the search using bisection
	while True:
		if min(v_2-v_1, v_0-v_1) < 1e-15:
			return c_1
		c_05 = 0.5 * (c_0 + c_1) 
		v_05 = invLF(c_05, e, V, p)
		if v_0 < v_05:
			return c_0
		if v_05 <= v_1:
			v_2 = v_1
			c_2 = c_1
			v_1 = v_05
			c_1 = c_05
		else:
			c_15 = 0.5 * (c_1 + c_2) 
			v_15 = invLF(c_15, e, V, p)
			if v_15 <= v_1:
				v_0 = v_1
				c_0 = c_1
				v_1 = v_15
				c_1 = c_15
			else:
				v_2 = v_15
				c_2 = c_15
				v_0 = v_05
				c_0 = c_05

def optimize_bound(e, c, V, p, projectionf, stepf, upper=True):
	''' Computes the upper or lower bound subject to constraints on the components of p.'''
	v_last = np.inf
	if np.isclose(e,0):
		def func(e,c,V,p):
			return p.dot(V)
	elif upper:
		def func(e,c,V,p):
			return p.dot(V) + (cCGF(c,V,p)+e)/c
	else:
		def func(e,c,V,p):
			return p.dot(V) - (cCGF(-c,V,p)+e)/c
	# print('UPPER' if upper else 'LOWER')
	c_orig = c
	p_orig = p
	while True:
		p_old = p
		t = time()
		p = maximize_over_P(c, e, V, p_orig, projectionf, stepf, upper=upper)
		# print('optimized %r --> %r   (%f)' % (p_old, p, time()-t))
		
		c_old = c
		t = time()
		c = minimize_over_c(c_orig, e, V, p, upper=upper)
		# print('optimized %f --> %f   (%f)' % (c_old, c, time()-t))
		
		v = func(e,c,V,p)
		# print(v, c, c_orig, p, p_orig)
		if np.abs(v_last-v) < 1e-15 or v_last < v:
			# print()
			return v 
		v_last = v

def optimize_bound_known_p(e, c, V, p, upper=True):
	''' Computes an lower bound or upper bound on E_Q[f(Z)] given P.'''
	v_last = np.inf
	if np.isclose(e,0):
		def func(e,c,V,p):
			return p.dot(V)
	elif upper:
		def func(e,c,V,p):
			return p.dot(V) + (cCGF(c,V,p)+e)/c
	else:
		def func(e,c,V,p):
			return p.dot(V) - (cCGF(-c,V,p)+e)/c
	while True:
		c = minimize_over_c(c, e, V, p, upper=upper)
		v_c = func(e,c,V,p)
		
		if np.abs(v_last-v_c) < 1e-15 or v_last < v_c:
			print()
			return v_c 
		v_last = v_c


def compute_robustness_bounds_goodman(e, V, D, c0=1e-12, delta=0.05):
	''' Computes a lower bound and upper bound on E_Q[f(Z)] given samples of f(Z) ~ P.'''
	k = len(V)
	n = len(D)
	p = np.array([ sum(D==v)/n for v in V ])	
	pmins, pmaxs = goodman_bound(p, n, k, delta)
	def projectionf(G, p, pmins=pmins, pmaxs=pmaxs):
		return project_grad_component_bounds(G, p, pmins, pmaxs)
	def stepf(p, G, c, V, max_step=np.inf, pmins=pmins, pmaxs=pmaxs):
		return get_largest_step_component_bounds(p, G, c, V, pmins, pmaxs, max_step=max_step)
	ub = optimize_bound(e,c0,V,p,projectionf,stepf,upper=True)
	lb = optimize_bound(e,c0,V,p,projectionf,stepf,upper=False)
	return ub, lb

def compute_robustness_bounds_ev(e, V, p, evmin, evmax, c0=1e-12, delta=0.05):
	''' Computes a lower bound and upper bound on E_Q[f(Z)] given samples of f(Z) ~ P.'''
	k = len(V)
	def projectionf(G, p, V=V, evmin=evmin, evmax=evmax):
		return project_grad_ev_bounds(G, p, V, evmin=evmin, evmax=evmax)
	def stepf(p, G, c, V, max_step=np.inf, evmin=evmin, evmax=evmax):
		return get_largest_step_ev_bounds(p, G, c, V, evmin=evmin, evmax=evmax, max_step=max_step)
	ub = optimize_bound(e,c0,V,p,projectionf,stepf,upper=True)
	lb = optimize_bound(e,c0,V,p,projectionf,stepf,upper=False)
	return ub, lb

def compute_robustness_bounds_known_p(e, V, p, c0=1e-12):
	''' Computes a lower bound and upper bound on E_Q[f(Z)] given P.'''
	ub = optimize_bound_known_p(e,c0,V,p,upper=True)
	lb = optimize_bound_known_p(e,c0,V,p,upper=False)
	return ub, lb