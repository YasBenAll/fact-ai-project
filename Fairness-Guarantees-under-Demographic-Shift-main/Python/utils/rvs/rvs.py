import numpy as np
import itertools 
from scipy.stats   import norm, chi, t
from scipy.special import erf, erfinv
from scipy.stats import beta
from scipy import optimize

from time import time
from copy import copy, deepcopy


from utils.rvs import utils

import re
import warnings

from utils.rvs.discrete_robustness import *
from utils.rvs.utils import safediv, safeprod, safesum, is_iterable, COMPARATORS, replace_keywords, load_probabilities, strip_is_funcs, optimize_demographic_shift, get_marginal_region_vertices, is_ttest, is_hoeffding, check_bounds_vector, is_estimate,optimize_on_simplex, get_opt_is_estimate_function, get_opt_is_hoeffding_function, get_opt_is_ttest_function
from utils.rvs.parser import get_parser
from utils.rvs import expressions
from utils import TimerCollection, keyboard





class ConstraintManager():

	def parse(self, string, debug=False):
		string = replace_keywords(string,self.keywords)
		parser = get_parser(debug=False)
		return parser.parse(string)

	def __init__(self, constraints, trivial_bounds={}, keywords={}, importance_samplers={}, seed=None, demographic_variable=None, demographic_variable_values=[], demographic_marginals=[], known_demographic_terms=None, debug=False, timername='CM'):
		self._raw_constraints = constraints
		self._raw_trivial_bounds = trivial_bounds
		self.keywords = keywords
		parser = get_parser(debug=False)
		pconstraints, replacements = [], []
		for const in constraints:
			pconstraint, replacement = replace_keywords(const,keywords,include_replacements=True)
			replacement = { o:('%s' % parser.parse(r)) for o,r in replacement.items() }
			pconstraints.append(pconstraint)
			replacements.append(replacement)
		self._replacements = replacements
		self.constraint_strs = pconstraints
		self.n_constraints = len(pconstraints)
		self.constraint_exprs = [ parser.parse(c) for c in pconstraints ]
		self.identify_base_variables()

		self._importance_samplers = importance_samplers
		
		self._demographic_variable    = demographic_variable
		self._demographic_variable_values    = demographic_variable_values
		self._demographic_marginals   = demographic_marginals
		try:
			self._demographic_marginal_verts = None if demographic_variable is None else get_marginal_region_vertices(demographic_marginals)
		except:
			self._demographic_marginal_verts = None
		self._known_demographic_terms = known_demographic_terms

		self.values = { n:None for n in self.referenced_variables }
		self.trivial_bounds = { v:(-np.inf,np.inf) for v in self.referenced_variables }
		self.trivial_bounds.update(trivial_bounds)
		self.trivial_bounds.update({ ev.sample_set.expression.name:(0,1) for ev in self.expected_values.values() if isinstance(ev.sample_set.expression, expressions.ComparatorExpression)})

		self._random = np.random.RandomState(seed)
		self._seed = seed
		self._debug = debug

		self._tc = TimerCollection(timername)

	def without_is(self):
		constraints = [ strip_is_funcs(s) for s in self._raw_constraints ]
		return ConstraintManager(constraints, self._raw_trivial_bounds, self.keywords, {}, self._seed)

	@property
	def referenced_variables(self):
		variables = set([ *self.base_variables, *self.cond_variables, *self.is_variables])
		if not(self._demographic_variable is None):
			variables.add(self._demographic_variable.name)
		return variables

	def identify_base_variables(self):
		# Identifies unique identifier variables in all constraints by their name
		self.base_variables = set()
		self.base_variables_per_constraint = []
		self.cond_variables = set()
		self.cond_variables_per_constraint = []
		self.is_variables   = set()
		self.is_variables_per_constraint = []
		self.expected_values = dict()
		self.expected_values_per_constraint = []
		for E in self.constraint_exprs:
			bvars, cvars, ivars, evars = self._identify_base_vars(E)
			self.base_variables_per_constraint.append(bvars)
			self.base_variables = self.base_variables.union(bvars)
			self.cond_variables_per_constraint.append(cvars)
			self.cond_variables = self.cond_variables.union(cvars)
			self.is_variables_per_constraint.append(ivars)
			self.is_variables   = self.is_variables.union(ivars)
			self.expected_values_per_constraint.append(evars)
			self.expected_values.update(evars)

	@staticmethod
	def _identify_base_vars(E):
		# Recursively identifies unique variables in <E> by their name
		if isinstance(E, expressions.ExpectedValue):
			S, C, _, _ = ConstraintManager._identify_base_vars(E.sample_set)
			I = set() if (E._is_expr is None) else set([E._is_expr.name])
			return S, C, I, {E.name:E}
		if isinstance(E, expressions.SampleSet):
			S, C, I, B = ConstraintManager._identify_base_vars(E.expression)
			if not(E.condition is None):
				C, _, _, _ = ConstraintManager._identify_base_vars(E.condition)
			return S, C, I, B
		if isinstance(E, expressions.ComparatorExpression):
			Se, Ce, Ie, B  = ConstraintManager._identify_base_vars(E.variable)
			Sv, Cv, Iv, Bv = ConstraintManager._identify_base_vars(E.value)
			B.update(Bv)
			return Se.union(Sv), Ce.union(Cv), Ie.union(Iv), B
		if isinstance(E, expressions.VariableExpression):
			return set([E.name]), set(), set(), dict()
		if isinstance(E, expressions.ConstantExpression):
			return set(), set(), set(), dict()
		base_vars, cond_vars, is_vars, ev_vars = set(), set(), set(), dict()
		for _E in E._terms:
			S, C, I, B = ConstraintManager._identify_base_vars(_E)
			base_vars = base_vars.union(S)
			cond_vars = cond_vars.union(C)
			is_vars   = is_vars.union(I)
			ev_vars.update(B)
		return base_vars, cond_vars, is_vars, ev_vars

	def set_data(self, values):
		# Sets defined variables to have the values in <values>
		for n in self.referenced_variables:
			if n in values.keys():
				self.values[n] = values[n]

	def has_defined_values(self):
		# Returns True iff all defined variables have non-None values set
		return not(any( self.values[v] is None for v in self.referenced_variables ))



	def evaluate(self):
		# Computes the value of each constraint expression given data set by set_data()
		assert self.has_defined_values(), 'ConstraintManager.evaluate(): Undefined values %r' % [ k for k,v in self.values.items() if v is None ]
		return np.array([ self._evaluate(E) for E in self.constraint_exprs ])

	def _evaluate(self, E):
		# Recursively evaluates expression <E> using data set by set_data()
		if isinstance(E, expressions.ConstantExpression):
			return E.value
		if isinstance(E, expressions.ExpectedValue):
			t = time()
			with warnings.catch_warnings():
				warnings.filterwarnings('error')
				try:
					t = time()
					S = self._evaluate(E.sample_set)
					if self._debug:
						print('time spent evaluating S:', time()-t)
					if len(S) == 0:
						return np.nan

					if not(self._demographic_variable is None):
						if E.sample_set.condition is None:
							D = self._evaluate(self._demographic_variable)
							known_terms = self._known_demographic_terms[None]
						else:
							D = self._evaluate(self._demographic_variable)[self._evaluate(E.sample_set.condition)]
							known_terms = self._known_demographic_terms[E.sample_set.condition.name]
						contains_intervals = check_bounds_vector(self._demographic_marginals)
						assert not(contains_intervals), '_evaluate(): Not supported for interval-valued demographic shifts.'
						return is_estimate(S, D, self._demographic_marginals, known_terms)


					elif not(E._is_func is None):
						# Now that we know the mean is well-defined, compute it. To account for importance 
						# sampling weights, we first represent the sum as a set of (num, den, val) triplets,
						# where num and den encode the numerator and denominator of each importance weight,
						# and val is the sum of the samples, for each corresponding unique input to th IS function.
						print('triggered IS')
						iterable_num, iterable_den = False, False
						is_inputs = self._evaluate(E._is_expr)
						is_func = self._importance_samplers[E._is_func]

						C = E.sample_set.condition
						if not(C is None):
							is_inputs = is_inputs[self._evaluate(E.sample_set.condition)]
						
						unique_is_inputs = np.unique(is_inputs)
						try:
							nums, dens = is_func(unique_is_inputs)
						except TypeError:
							nums = []
							dens = []
							for v in unique_is_inputs:
								num, den = is_func(v)
								nums.append(num)
								dens.append(den)
								iterable_num = iterable_num or is_iterable(num)
								iterable_den = iterable_den or is_iterable(den)

						vals = [ np.mean(S[is_inputs==v]) for v in unique_is_inputs ]
							
						# Now we *might* compute the actual mean. If all of the numerators and denominators are
						# known, then this s the weighted sums of (n/d)*v for each (n,d,v) triplet.
						# However, it's possible that n or d can be a range, representing uncertainty in the 
						# numerators or denominators. Thus, we cannot assign a unique value to the mean.
						assert not(iterable_num or iterable_den), 'Cannot compute a single value for range-valued importance sampling weights.'
						mean = 0.0		
						for num,den,val in zip(nums, dens, vals):
							mean += (num/den) * val
						return mean
					else:
						# No demographic shift or importance sampling
						return np.mean(S)
				except RuntimeWarning:
					return np.nan
		if isinstance(E, expressions.SampleSet):
			values = self._evaluate(E.expression)
			if not(E.condition is None):
				cvalues = self._evaluate(E.condition)
				values = values[cvalues]
			return values
		if isinstance(E, expressions.ComparatorExpression):
			values1 = self._evaluate(E.variable)
			values2 = self._evaluate(E.value)
			return COMPARATORS[E.comparator](values1, values2)
		if isinstance(E, expressions.VariableExpression):
			if E._special == 'index':
				return np.arange(len(self.values[E.name]))
			return self.values[E.name]
		if isinstance(E, expressions.NegativeExpression):
			return -self._evaluate(E._terms[0])
		if isinstance(E, expressions.NotExpression):
			return not(self._evaluate(E._terms[0]))
		if isinstance(E, expressions.AbsExpression):
			return np.abs(self._evaluate(E._terms[0]))
		if isinstance(E, expressions.SumExpression):
			return np.sum([ self._evaluate(_E) for _E in E._terms ])
		if isinstance(E, expressions.ProductExpression):
			return np.prod([ self._evaluate(_E) for _E in E._terms ])
		if isinstance(E, expressions.AndExpression):
			values = self._evaluate(E._terms[0])
			for _E in E._terms[1:]:
				values = np.logical_and(values, self._evaluate(_E))
			return values
		if isinstance(E, expressions.OrExpression):
			values = self._evaluate(E._terms[0])
			for _E in E._terms[1:]:
				values = np.logical_or(values, self._evaluate(_E))
			return values
		if isinstance(E, expressions.FractionExpression):
			v_num = self._evaluate(E._terms[0]) 
			v_den = self._evaluate(E._terms[1])
			return safediv(v_num, v_den)
		if isinstance(E, expressions.MaxExpression):
			return np.max([ self._evaluate(_E) for _E in E._terms ])
		if isinstance(E, expressions.NANMaxExpression):
			return np.nanmax([ self._evaluate(_E) for _E in E._terms ])
		if isinstance(E, expressions.MaxRecipExpression):
			_E = E._terms[0]
			if isinstance(_E, expressions.FractionExpression):
				v_num = self._evaluate(_E._terms[0])
				v_den = self._evaluate(_E._terms[1])
				vs = [safediv(v_num, v_den), safediv(v_den, v_num)]
			else:
				v  = self._evaluate(_E)
				vs = [v, safediv(1, v)]
			if all(np.isnan(vs)):
				return np.nan
			return np.nanmax(vs)

	def upper_bound_constraints(self, all_deltas, mode='hoeffding', interval_scaling=1.0, n_scale=1.0, robustness_bounds={}, term_values={}, debug=False):
		all_deltas = np.zeros(self.n_constraints) if all_deltas is None else all_deltas
		self._tc.tic('bound_constraints')
		constraint_bounds = self.bound_constraints(all_deltas, mode=mode, interval_scaling=interval_scaling, n_scale=n_scale, robustness_bounds=robustness_bounds, term_values=term_values)
		if debug:
			print('CBs:', constraint_bounds)
		self._tc.toc('bound_constraints')
		out = np.array([ b[1] for b in constraint_bounds ])
		if debug:		
			print('-'*40)
			self._tc.print_total_times()
			self._tc.print_avg_times()
			self._tc.print_num_tics()
			print('-'*40)
		return out

	def lower_bound_constraints(self, all_deltas, mode='hoeffding', interval_scaling=1.0, n_scale=1.0, robustness_bounds={}, term_values={}, debug=True):
		all_deltas = np.zeros(self.n_constraints) if all_deltas is None else all_deltas
		constraint_bounds = self.bound_constraints(all_deltas, mode=mode, interval_scaling=interval_scaling, n_scale=n_scale, robustness_bounds=robustness_bounds, term_values=term_values)
		if debug:
			print('CBs:', constraint_bounds)
		return np.array([ b[0] for b in constraint_bounds ])

	def bound_constraints(self, all_deltas, mode='hoeffding', interval_scaling=1.0, n_scale=1.0, robustness_bounds={}, term_values={}):
		self._tc.tic('bound_constraints:setup')
		assert self.has_defined_values(), 'ConstraintManager.bound(): Undefined values %r' % [ k for k,v in self.values.items() if v is None ]
		deltas   = { name : None for name in self.expected_values }
		bounds   = { name : None for name in self.expected_values }
		epsilons = { name : None for name in self.expected_values }
		constraint_bounds = []
		# Decode robustness bounds
		robustness_bounds = { self.parse(replace_keywords(term,self.keywords)).name : epsilon for term,epsilon in robustness_bounds }
		self._tc.toc('bound_constraints:setup')

		for cnum, (E,delta_tot) in enumerate(zip(self.constraint_exprs,all_deltas)):
			# Bound the base variables needed for this constraint
			variables = self.expected_values_per_constraint[cnum]

			if len(variables) > 0:
				delta_per_var = delta_tot / len(variables)			
				for name in variables:
					epsilon = robustness_bounds[name] if name in robustness_bounds.keys() else 0.0
					# if not(deltas[name] is None) and (deltas[name] == delta_per_var) and (epsilons[name] == epsilon):
					# 	bounds[name] = bounds_last[name]
					# 	# print('      Bounds for %s are' % name, bounds[name])
					# else:
					self._tc.tic(f'bound_constraints[{cnum}]:bound_variable:{name}')
					bounds[name] = self.bound_variable(name, delta_per_var, mode=mode, n_scale=n_scale)
					self._tc.toc(f'bound_constraints[{cnum}]:bound_variable:{name}')
					# print('      Bounds for %s are' % name, bounds[name])
					self._tc.tic(f'bound_constraints[{cnum}]:robustness_bounds:{name}')
					if not(np.isclose(epsilon,0)):
						evmin, evmax = bounds[name]
						sample_set = self.parse(name).sample_set
						S = np.array(self._evaluate(sample_set), dtype=float)
						# Determine the set of values that samples for this EV can take on
						if isinstance(sample_set.expression, (expressions.AndExpression, expressions.OrExpression, expressions.ComparatorExpression)):
							V = np.array([0, 1])
						elif (name in term_values.keys()):
							V = term_values[name]
						else:
							raise RuntimeWarning('Values have not been provided for term \'%s\'. Basing V off of unique values encountered in %d samples, which may be incomplete.' % (name,len(S)))
							V = np.unique(S)
						
						if len(S) == 0:
							ub, lb = V.max(), V.min()
						else:
							p = np.array([ np.mean(S==v) for v in V ])
							ub, lb = compute_robustness_bounds_ev(epsilon, V, p, evmin, evmax)
						bounds[name] = (lb, ub)
					self._tc.tic(f'bound_constraints[{cnum}]:robustness_bounds:{name}')
					deltas[name] = delta_per_var
					epsilons[name] = epsilon
			# Bound the expression for this constraint
			self._tc.tic(f'bound_constraints[{cnum}]:bound_expression')
			l, u = ConstraintManager.bound_expression(E, bounds={v:bounds[v] for v in variables})
			self._tc.toc(f'bound_constraints[{cnum}]:bound_expression')
			# Inflate the bound if needed
			if not(any(np.isinf([l,u])) or any(np.isnan([l,u]))):
				mod = 0.5*(u-l)*(interval_scaling-1)
				l, u = l-mod, u+mod
			constraint_bounds.append((l,u))
		return constraint_bounds

	def bound_expression(E, bounds):
		if isinstance(E, expressions.ConstantExpression):
			return (E.value, E.value)
		if isinstance(E, expressions.ExpectedValue):
			return bounds[E.name]
		if isinstance(E, expressions.NegativeExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			return (-u, -l)
		if isinstance(E, expressions.NotExpression):
			return (0, 1)
			v = 1 - ConstraintManager.bound_expression(E._terms[0], bounds)
			return (v, v)
		if isinstance(E, expressions.AbsExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			if l <= 0 and u >= 0:
				return (0, max(u,-l))
			if u < 0:
				return (-u, -l)
			if l > 0:
				return (l, u)
		if isinstance(E, expressions.SumExpression):
			l, u = 0, 0
			for _E in E._terms:
				_l, _u = ConstraintManager.bound_expression(_E, bounds)
				l, u = safesum(l,_l), safesum(u,_u)
			return (l, u)
		if isinstance(E, expressions.ProductExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			for _E in E._terms[1:]:
				ln, un = ConstraintManager.bound_expression(_E, bounds)
				cll, clu = safeprod(l,ln), safeprod(l,un)
				cul, cuu = safeprod(u,ln), safeprod(u,un)
				interval_corners = [cll, clu, cul, cuu]
				l, u = min(interval_corners), max(interval_corners)
			return (l,u)
		if isinstance(E, expressions.FractionExpression):
			ln, un = ConstraintManager.bound_expression(E._terms[0], bounds)
			ld, ud = ConstraintManager.bound_expression(E._terms[1], bounds)
			# If ln = un = ld = ud = 0, we return (-inf,inf) which is a useless bound, instead of (nan,nan)
			# Values are based on treating the input intervals as open intervals
			# If an interval is empty (l==u), it is treated as an infinitesimal interval (-e+l,l+e) instead
			if (ld == 0) and (ud == 0):
				return (-np.inf, np.inf)
			if (ld == 0 or ud == 0) and (ln <= 0) and (un >= 0):
				return (-np.inf, np.inf)
			if (ld == 0) and (ln == 0):
				return (np.inf, np.inf)
			if (ld == 0 or ud == 0) and (un <= 0):
				return (np.inf, np.inf)
			cll, clu = safediv(ln,ld), safediv(ln,ud)
			cul, cuu = safediv(un,ld), safediv(un,ud)
			interval_corners = [cll, clu, cul, cuu]
			return min(interval_corners), max(interval_corners)
		if isinstance(E, expressions.MaxExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			for _E in E._terms[1:]:
				ln, un = ConstraintManager.bound_expression(_E, bounds)
				l, u = max(l,ln), max(u,un)
			return (l,u)
		if isinstance(E, expressions.MaxExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			for _E in E._terms[1:]:
				ln, un = ConstraintManager.bound_expression(_E, bounds)
				l, u = np.nanmax([l,ln]), np.nanmax([u,un])
			return (l,u)
		if isinstance(E, expressions.MaxRecipExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			li = 1/l if not(l==0) else np.inf
			ui = 1/u if not(u==0) else np.inf

			if any(np.isnan([l,u])):
				return (np.nan, np.nan)

			elif l >= 1 and u >= 1:
				return (l, u)
			elif l >= 0 and u >= 1:
				return (1, max(u,li))
			elif l >= -1 and u >= 1:
				return (l, np.inf)
			elif l < -1 and u >= 1:
				return (-1, np.inf)

			elif l >= 0 and u >= 0:
				return (ui, li)
			elif l >= -1 and u >= 0:
				return (l, np.inf)
			elif l < -1 and u >= 0:
				return (-1, np.inf)

			elif l >= -1 and u >= -1:
				return (l, u)
			elif l < -1 and u >= -1:
				return (-1, max(li, u))

			elif l < -1 and u < -1:
				return (ui, li)

	def bound_variable(self, name, delta, mode='hoeffding', n_scale=1.0, bootstrap_samples=1000):
		# Returns a <delta>-probability confidence interval on the value of <name> using <mode>.	
		self._tc.tic(f'bound_constraints:bound_variable:{name}:setup')
		E = self.expected_values[name] 
		
		mode = mode.lower()
		if isinstance(E.sample_set.expression, (expressions.AndExpression,expressions.OrExpression)):
			a, b = (0,1)
		else:
			a, b = self.trivial_bounds[E.sample_set.expression.name]

		# "Computes" the trivial bounds
		if mode == 'trivial':
			return (a, b)
		self._tc.toc(f'bound_constraints:bound_variable:{name}:setup')

		# Get the sample(s) associated with <name>
		self._tc.tic(f'bound_constraints:bound_variable:{name}:evaluate')
		S = self._evaluate(E.sample_set)
		try:
			n = len(S)
		except TypeError:
			S = np.array([ S ])
			n = len(S)
		self._tc.toc(f'bound_constraints:bound_variable:{name}:evaluate')
		
		if (n == 0 or n_scale == 0):
			return (a, b)

		if not(self._demographic_variable is None):
			self._tc.tic(f'bound_constraints:bound_variable:{name}:bound_ds')
			val = self._bound_variable_with_demographic_shift(name, E, S, n, delta, a, b, mode=mode, n_scale=n_scale)
			self._tc.toc(f'bound_constraints:bound_variable:{name}:bound_ds')
			return val
		if not(E._is_func is None):
			self._tc.tic(f'bound_constraints:bound_variable:{name}:bound_is')
			val = self._bound_variable_with_importance_sampling(E, S, n, delta, a, b, mode=mode, n_scale=n_scale)
			self._tc.toc(f'bound_constraints:bound_variable:{name}:bound_is')
			return val
		else:
			self._tc.tic(f'bound_constraints:bound_variable:{name}:bound_normal')
			val = self._bound_variable_standard(E, S, n, delta, a, b, mode=mode, n_scale=n_scale)
			self._tc.toc(f'bound_constraints:bound_variable:{name}:bound_normal')
			return val
		

	def _bound_variable_with_demographic_shift(self, name, E, S, n, delta, a, b, mode='hoeffding', n_scale=1.0, bootstrap_samples=1000):
		self._tc.tic(f'bound_constraints:bound_variable:{name}:bound_ds:evaluate')
		if E.sample_set.condition is None:
			D = self._evaluate(self._demographic_variable)
			known_terms = self._known_demographic_terms[None]
		else:
			D = self._evaluate(self._demographic_variable)[self._evaluate(E.sample_set.condition)]		
			known_terms = self._known_demographic_terms[E.sample_set.condition.name]
		Q_D = self._demographic_marginals
		self._tc.toc(f'bound_constraints:bound_variable:{name}:bound_ds:evaluate')
		self._tc.tic(f'bound_constraints:bound_variable:{name}:bound_ds:check')
		contains_intervals = check_bounds_vector(Q_D)
		self._tc.toc(f'bound_constraints:bound_variable:{name}:bound_ds:check')

		# If mode is none, then don't construct confidence intervals, and use intervals based on pointwise estimates
		if (mode is None) or (mode.lower() == 'none'):
			self._tc.tic(f'bound_constraints:bound_variable:{name}:bound_ds:optimize_estimate')
			if contains_intervals:
				# f = lambda _q: -is_estimate(S, D, _q, known_terms=known_terms)
				# g = lambda _q:  is_estimate(S, D, _q, known_terms=known_terms)
				optf = get_opt_is_estimate_function(S,D,known_terms=known_terms)
				f = lambda _q: -optf(_q)
				g = lambda _q:  optf(_q)
				result_f = optimize_on_simplex(f, Q_D)
				result_g = optimize_on_simplex(g, Q_D)
				ub = is_estimate(S, D, result_f, known_terms=known_terms)
				lb = is_estimate(S, D, result_g, known_terms=known_terms)
			else:
				lb = is_estimate(S, D, Q_D, known_terms=known_terms)
				ub = lb
			self._tc.toc(f'bound_constraints:bound_variable:{name}:bound_ds:optimize_estimate')
			return (max(a,lb), min(b,ub))

		# Computes the hoeffding bound
		if mode == 'hoeffding':
			self._tc.tic(f'bound_constraints:bound_variable:{name}:bound_ds:optimize_hoeffding')
			if contains_intervals:
				# f = lambda _q: -is_hoeffding(S, D, delta, _q, a=a, b=b, known_terms=known_terms, n_scale=n_scale)[1]
				# g = lambda _q:  is_hoeffding(S, D, delta, _q, a=a, b=b, known_terms=known_terms, n_scale=n_scale)[0]
				optf = get_opt_is_hoeffding_function(S,D,delta,known_terms=known_terms,n_scale=n_scale)
				f = lambda _q: -optf(_q)[1]
				g = lambda _q:  optf(_q)[0]
				result_f = optimize_on_simplex(f, Q_D)
				result_g = optimize_on_simplex(g, Q_D)
				ub = is_hoeffding(S, D, delta, result_f, known_terms=known_terms, n_scale=n_scale)[1]
				lb = is_hoeffding(S, D, delta, result_g, known_terms=known_terms, n_scale=n_scale)[0]
			else:
				lb, ub = is_hoeffding(S, D, delta, Q_D, a=a, b=b, known_terms=known_terms, n_scale=n_scale)
			self._tc.toc(f'bound_constraints:bound_variable:{name}:bound_ds:optimize_hoeffding')
			return (max(a,lb), min(b,ub))

		# Computes the bootstrap bound
		if mode == 'bootstrap':
			raise RuntimeError('Bootstrap bounds are not supported with demographic shift currently.')

		#### Bounds below this point require more than two samples ####
		
		if len(S) <= 2:
			return (a, b)

		# Computes the t-test inversion bound 	
		if mode == 'ttest': 
			self._tc.tic(f'bound_constraints:bound_variable:{name}:bound_ds:optimize_ttest')
			if contains_intervals:
				# f = lambda _q: -is_ttest(S, D, delta, _q, known_terms=known_terms, n_scale=n_scale)[1]
				# g = lambda _q:  is_ttest(S, D, delta, _q, known_terms=known_terms, n_scale=n_scale)[0]
				optf = get_opt_is_ttest_function(S,D,delta,known_terms=known_terms,n_scale=n_scale)
				f = lambda _q: -optf(_q)[1]
				g = lambda _q:  optf(_q)[0]
				Q_D = [(1e-8,1.0)] * len(Q_D) if Q_D is None else Q_D
				result_f = optimize_on_simplex(f, Q_D)
				result_g = optimize_on_simplex(g, Q_D)
				ub = is_ttest(S, D, delta, result_f, known_terms=known_terms, n_scale=n_scale)[1]
				lb = is_ttest(S, D, delta, result_g, known_terms=known_terms, n_scale=n_scale)[0]
			else:
				lb, ub = is_ttest(S, D, delta, Q_D, known_terms=known_terms, n_scale=n_scale)
			self._tc.toc(f'bound_constraints:bound_variable:{name}:bound_ds:optimize_ttest')
			return (max(a,lb), min(b,ub))

		# Should never reach here, so return trivial bounds
		return (a, b)


	def _bound_variable_with_importance_sampling(self, E, S, n, delta, a, b, mode='hoeffding', n_scale=1.0, bootstrap_samples=1000):
		n_scaled = n * n_scale	


		# Now that we know the mean is well-defined, compute it. To account for importance 
		# sampling weights, we first represent the sum as a set of (num, den, val) triplets,
		# where num and den encode the numerator and denominator of each importance weight,
		# and val is the sum of the samples, for each corresponding unique input to th IS function.
		iterable_num, iterable_den = False, False
		is_inputs = self._evaluate(E._is_expr)

		C = E.sample_set.condition
		if not(C is None):
			is_inputs = is_inputs[self._evaluate(E.sample_set.condition)]
		is_func = self._importance_samplers[E._is_func]
		unique_is_inputs = np.unique(is_inputs)
		nums = []
		dens = []
		for v in unique_is_inputs:
			num, den = is_func(v)
			nums.append(num)
			dens.append(den)
			iterable_num = iterable_num or is_iterable(num)
			iterable_den = iterable_den or is_iterable(den)
		vals = [ np.sum(S[is_inputs==v])/n for v in unique_is_inputs ]


		# Computes the hoeffding bound
		if mode == 'hoeffding':
			if not(iterable_num or iterable_den):
				mean = 0.0
				for num,den,val in zip(nums, dens, vals):
					mean += (num/den) * val
				l, u = mean, mean
			else:
				assert not(iterable_den), 'Importance weights with denominator ranges are not yet supported for Hoeffding-based bounds.'
				V = vals / np.array(dens)
				B = np.array([ [b[0],b[1]] if is_iterable(b) else [b,b] for b in nums ])
				l = load_probabilities(B,V,ascending=True).dot(V)
				u = load_probabilities(B,V,ascending=False).dot(V)
			offset = (b-a) * np.sqrt(0.5*np.log(2/delta)/n_scaled)
			l, u = safesum(l,-offset), safesum(u,offset)
			return (max(a,l), min(b,u))

		# Computes the bootstrap bound
		if mode == 'bootstrap':
			C = self._random.multinomial(n, np.ones(n)/n, bootstrap_samples)
			B = np.array([ [b[0],b[1]] if is_iterable(b) else [b,b] for b in nums ])
			lbs, ubs = [], []
			for j,_C in enumerate(C):
				_S = _C * S
				vals = [ np.sum(_S[is_inputs==v])/n for v in unique_is_inputs ]
				V = vals / np.array(dens)
				l = load_probabilities(B,V,ascending=True).dot(V)
				u = load_probabilities(B,V,ascending=False).dot(V)
				lbs.append(l)
				ubs.append(u)
			l = np.percentile(lbs, 50*delta)
			u = np.percentile(ubs, 100 - 50*delta)
			return (max(a,l), min(b,u))

		#### Bounds below this point require at least two samples ####
		
		if len(S) == 1:
			return (a, b)

		# Computes the t-test inversion bound 	
		if mode == 'ttest': 
			if not(iterable_num or iterable_den):
				mean = 0.0
				for num,den,val in zip(nums, dens, vals):
					mean += (num/den) * val
				# TODO THIS IS WRONG. DS ISNT INCORPORATED INTO THE STD TERM 
				assert False, 'Importance weights are not yet supported for t-Test-based bounds.'
				# Now that we know the standard deviation is well-defined, compute it
				std = np.std(S,ddof=1)
				# If standard deviation too close to zero, apply the rule of three
				if np.isclose(std, 0.0):
					if np.isclose(mean, a):
						return (a, (b-a)*3.0/n_scaled)
					elif np.isclose(mean, b):
						return ((b-a)*(1-3.0/n_scaled)+a, b)
					return (a, b)
				
				offset = std * t.ppf(1-delta/2,n-1) / np.sqrt(n-1)
				l, u = safesum(mean,-offset), safesum(mean,offset)
			else:
				assert False, 'Importance weights are not yet supported for t-Test-based bounds.'
			
			return (max(a,l), min(b,u))






	def _bound_variable_standard(self, E, S, n, delta, a, b, mode='hoeffding', n_scale=1.0, bootstrap_samples=1000):
		n_scaled = n * n_scale	

		# Computes the hoeffding bound
		if mode == 'hoeffding':
			mean = np.mean(S)
			offset = (b-a) * np.sqrt(0.5*np.log(2/delta)/n_scaled)
			l, u = safesum(mean,-offset), safesum(mean,offset)
			return (max(a,l), min(b,u))

		# Computes the bootstrap bound
		if mode == 'bootstrap':
			C = self._random.multinomial(n, np.ones(n)/n, bootstrap_samples)
			vals = [ np.mean(_C*S) for _C in C ]
			l = np.percentile(vals, 50*delta)
			u = np.percentile(vals, 100 - 50*delta)
			return (max(a,l), min(b,u))

		#### Bounds below this point require at least two samples ####
		
		if len(S) == 1:
			return (a, b)

		# Computes the t-test inversion bound 	
		if mode == 'ttest': 
			mean = np.mean(S)
			# Now that we know the standard deviation is well-defined, compute it
			std = np.std(S,ddof=1)
			# If standard deviation too close to zero, apply the rule of three
			if np.isclose(std, 0.0):
				if np.isclose(mean, a):
					return (a, (b-a)*3.0/n_scaled)
				elif np.isclose(mean, b):
					return ((b-a)*(1-3.0/n_scaled)+a, b)
				return (a, b)
			offset = std * t.ppf(1-delta/2,n-1) / np.sqrt(n-1)
			l, u = safesum(mean,-offset), safesum(mean,offset)
			return (max(a,l), min(b,u))
