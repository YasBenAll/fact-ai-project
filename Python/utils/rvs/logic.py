import numpy as np
from utils.rvs import expressions
from utils.rvs.expressions import RVFuncs
from utils.rvs.utils import COMPARATOR_NEGATIONS
from matplotlib import pyplot as plt

def convert_to_DNF(E, level=0):
	if isinstance(E, expressions.AndExpression):
		and_terms = []
		for _term in [ convert_to_DNF(_E, level+1) for _E in E._terms ]:
			if len(and_terms) == 0:
				and_terms = _term._terms
			else:
				and_terms = [ expressions.AndExpression(t0._terms + t1._terms) for t0 in and_terms for t1 in _term._terms ]
		return expressions.OrExpression(and_terms)

	elif isinstance(E, expressions.OrExpression):
		or_terms = []
		for _term in [ convert_to_DNF(_E, level+1) for _E in E._terms ]:
			or_terms.extend(_term._terms)
		return expressions.OrExpression(or_terms)
	else:
		return expressions.OrExpression([ expressions.AndExpression([ E ]) ])


def half_interval(expr, c, v):
	return RVFuncs.comparator_variable(expr, c, RVFuncs.constant(v))

def interval(expr, c1, v1, c2, v2):
	return expressions.AndExpression([ half_interval(expr, c1, v1), half_interval(expr, c2, v2) ])

def intervals(expr, *interval_tuples):
	terms = []
	for interval_tuple in interval_tuples:
		if len(interval_tuple) == 2:
			terms.append(half_interval(expr, *interval_tuple))
		elif len(interval_tuple) == 4:
			terms.append(interval(expr, *interval_tuple))
		else:
			raise RuntimeException('intervals(): interval tuple does not have the correct number of terns: %r' % interval_tuple)
	return terms #, expressions.OrExpression(terms)


# def process_interval(E):
# 	if isinstance(E, expressions.AndExpression):
# 		assert len(E._terms) == 2, 'E has more than two terms.'
# 		assert all([ isinstance(t,expressions.ComparatorExpression) for t in E._terms ]), 'E contains non-comparison terms.'
# 		a, b = E._terms
# 		if a.value.value > b.value.value:
# 			a, b = b, a
# 		assert (a.comparator in ['>','>='] and b.comparator in ['<','<=']), 'E does not form an interval.'
# 		return a.comparator, a.value.value, b.comparator, b.value.value
# 	elif isinstance(E, expressions.ComparatorExpression):
# 		c, v = E.comparator, E.value.value
# 		assert not(c == '='), 'E cannot be !=.'
# 		if c in ['>','>=']:
# 			return c, v, '<', np.inf
# 		elif c in ['<','<=']:
# 			return '>', -np.inf, c, v
# 		else:
# 			return '=', v, '=', v




def process_interval(E):
	if isinstance(E, expressions.AndExpression):
		assert len(E._terms) == 2, 'E has more than two terms.'
		assert all([ isinstance(t,expressions.ComparatorExpression) for t in E._terms ]), 'E contains non-comparison terms.'
		a, b = E._terms
		if a.value.value > b.value.value:
			a, b = b, a
		assert (a.comparator in ['>','>='] and b.comparator in ['<','<=']), 'E does not form an interval.'
		return a, b
	elif isinstance(E, expressions.ComparatorExpression):
		c, v = E.comparator, E.value.value
		assert not(c == '!='), 'E cannot be !=.'
		if c in ['>','>=']:
			return E, None
		elif c in ['<','<=']:
			return None, E
		else:
			return RVFuncs.comparator_variable(E.variable, '>=', E.value), RVFuncs.comparator_variable(E.variable, '<=', E.value)

def partition_intervals(E1, E2):
	if isinstance(E1,expressions.ComparatorExpression) and isinstance(E2,expressions.ComparatorExpression):
		return partition(E1,E2)

	a1, b1 = process_interval(E1)
	a2, b2 = process_interval(E2)

	if not(a1 is None) and not(a2 is None):
		a1a2 = partition(a1,a2)[1][0]
	else:
		a1a2 = a1 if (a2 is None) else a2

	if not(b1 is None) and not(b2 is None):
		b1b2 = partition(b1,b2)[1][0]
	else:
		b1b2 = b1 if (b2 is None) else b2
		
	both = partition(a1a2, b1b2)[1]
	if len(both) == 0:
		return [E1], [], [E2]

	not_a1a2 = RVFuncs.logical_not(a1a2)
	not_b1b2 = RVFuncs.logical_not(b1b2)
	E1_only = (partition(a1, not_a1a2)[1] if not(a1 is None) else []) + \
			  (partition(b1, not_b1b2)[1] if not(b1 is None) else [])
	E2_only = (partition(a2, not_a1a2)[1] if not(a2 is None) else []) + \
			  (partition(b2, not_b1b2)[1] if not(b2 is None) else [])
	
	return E1_only, both, E2_only


def intersect_expressions(E_list):
	terms = []
	for _term in E_list:
		if isinstance(_term, expressions.ComparatorExpression) and (_term.comparator == '!='):
			E1 = RVFuncs.comparator_variable(_term.variable, '<', _term.value)
			E2 = RVFuncs.comparator_variable(_term.variable, '>', _term.value)
			if len(terms) == 0:
				terms = [ E1, E2 ]
			else:
				new_terms = []
				for term in terms:
					A1 = partition_intervals(term,E1)[1]
					A2 = partition_intervals(term,E2)[1]
					if len(A1) > 0:
						new_terms.extend(A1)
					if len(A2) > 0:
						new_terms.extend(A2)
				terms = new_terms
		else:
			if len(terms) == 0:
				terms = [_term]
			else:
				new_terms = []
				for term in terms:
					A = partition_intervals(term,_term)[1]
					if len(A) > 0:
						new_terms.extend(A)
				terms = new_terms
	return terms


def partition(E1, E2):
	assert isinstance(E1, expressions.ComparatorExpression) and isinstance(E2, expressions.ComparatorExpression), 'Inputs must be ComparatorExpression objects.'
	assert E1.variable == E2.variable, 'Inputs must refer to the same variable.'

	expr = E1.variable
	c1, v1 = E1.comparator, E1.value.value
	c2, v2 = E2.comparator, E2.value.value

	I = lambda *interval_tuples: intervals(expr, *interval_tuples)

	if (c1=='<') and (c2=='<'):
		if (v1 < v2):
			return [], I(('<',v1)), I(('>=',v1,'<',v2))
		if (v1 == v2):
			return [], I(('<', v1)), []
		if (v1 > v2):
			return I(('>=',v2,'<',v1)), I(('<',v2)), []

	if (c1=='<') and (c2=='>'):
		if (v1 < v2):
			return I(('<',v1)), [], I(('>',v2))
		if (v1 == v2):
			return I(('<',v1)), [], I(('>',v2))
		if (v1 > v2):
			return I(('<=',v2)), I(('>',v2,'<',v1)), I(('>=',v1))

	if (c1=='<') and (c2=='<='):
		if (v1 < v2):
			return [], I(('<',v1)), I(('>=',v1,'<=',v2))
		if (v1 == v2):
			return [], I(('<',v1)), I(('=',v2))
		if (v1 > v2):
			return I(('>',v2,'<',v1)), I(('<=',v2)), []

	if (c1=='<') and (c2=='>='):
		if (v1 < v2):
			return I(('<',v1)), [], I(('>=',v2))
		if (v1 == v2):
			return I(('<',v1)), [], I(('>=',v2))
		if (v1 > v2):
			return I(('<',v2)), I(('>=',v2,'<',v1)), I(('>=',v1))

	if (c1=='<') and (c2=='='):
		if (v1 < v2):
			return I(('<',v1)), [], I(('=',v2))
		if (v1 == v2):
			return I(('<',v1)), [], I(('=',v2))
		if (v1 > v2):
			return I(('<',v2),('>',v2,'<',v1)), I(('=',v2)), []

	if (c1=='<') and (c2=='!='):
		if (v1 < v2):
			return [], I(('<',v1)), I(('>=',v1,'<',v2),('>',v2))
		if (v1 == v2):
			return [], I(('<',v1)), I(('>',v2))
		if (v1 > v2):
			return I(('=',v2)), I(('<',v2),('>',v2,'<',v1)), I(('>=',v1))


	if (c1=='>') and (c2=='<'):
		if (v1 < v2):
			return I(('>=',v2)), I(('>',v1,'<',v2)), I(('<=',v1))
		if (v1 == v2):
			return I(('>',v1)), [], I(('<',v2))
		if (v1 > v2):
			return I(('>',v1)), [], I(('<',v2))

	if (c1=='>') and (c2=='>'):
		if (v1 < v2):
			return I(('>',v1,'<=',v2)), I(('>',v2)), []
		if (v1 == v2):
			return [], I(('>',v1)), []
		if (v1 > v2):
			return [], I(('>',v1)), I(('>',v2,'<=',v1))

	if (c1=='>') and (c2=='<='):
		if (v1 < v2):
			return I(('>',v2)), I(('>',v1,'<=',v2)), I(('<=',v1))
		if (v1 == v2):
			return I(('>',v1)), [], I(('<=',v2))
		if (v1 > v2):
			return I(('>',v1)), [], I(('<=',v2))

	if (c1=='>') and (c2=='>='):
		if (v1 < v2):
			return I(('>',v1,'<',v2)), I(('>=',v2)), []
		if (v1 == v2):
			return [], I(('>',v1)), I(('=',v1))
		if (v1 > v2):
			return [], I(('>',v1)), I(('>=',v2,'<=',v1))

	if (c1=='>') and (c2=='='):
		if (v1 < v2):
			return I(('>',v1,'<',v2),('>',v2)), I(('=',v2)), []
		if (v1 == v2):
			return I(('>',v1)), [], I(('=',v2))
		if (v1 > v2):
			return I(('>',v1)), [], I(('=',v2))

	if (c1=='>') and (c2=='!='):
		if (v1 < v2):
			return I(('=',v2)), I(('>',v1,'<',v2),('>',v2)), I(('<=',v1))
		if (v1 == v2):
			return [], I(('>',v1)), I(('<',v2))
		if (v1 > v2):
			return [], I(('>',v1)), I(('<',v2),('>',v2,'<=',v1))



	if (c1=='<=') and (c2=='<'):
		if (v1 < v2):
			return [], I(('<=',v1)), I(('>',v1,'<',v2))
		if (v1 == v2):
			return I(('=',v1)), I(('<',v1)), []
		if (v1 > v2):
			return I(('>=',v2,'<=',v1)), I(('<',v2)), []

	if (c1=='<=') and (c2=='>'):
		if (v1 < v2):
			return I(('<=',v1)), [], I(('>',v2))
		if (v1 == v2):
			return I(('<=',v1)), [], I(('>',v2))
		if (v1 > v2):
			return I(('<=',v2)), I(('>',v2,'<=',v1)), I(('>',v1))

	if (c1=='<=') and (c2=='<='):
		if (v1 < v2):
			return [], I(('<=',v1)), I(('>',v1,'<=',v2))
		if (v1 == v2):
			return [], I(('<=',v1)), []
		if (v1 > v2):
			return I(('>',v2,'<=',v1)), I(('<=',v2)), []

	if (c1=='<=') and (c2=='>='):
		if (v1 < v2):
			return I(('<=',v1)), [], I(('>=',v2))
		if (v1 == v2):
			return I(('<',v1)), I(('=',v1)), I(('>',v1))
		if (v1 > v2):
			return I(('<',v2)), I(('>=',v2,'<=',v1)), I(('>',v1))

	if (c1=='<=') and (c2=='='):
		if (v1 < v2):
			return I(('<=',v1)), [], I(('=',v2))
		if (v1 == v2):
			return I(('<',v1)), I(('=',v2)), []
		if (v1 > v2):
			return I(('<',v2),('>',v2,'<=',v1)), I(('=',v2)), []

	if (c1=='<=') and (c2=='!='):
		if (v1 < v2):
			return [], I(('<=',v1)), I(('>',v1,'<',v2),('>',v2))
		if (v1 == v2):
			return I(('=',v1)), I(('<',v1)), I(('>',v1))
		if (v1 > v2):
			return I(('=',v2)), I(('<',v2), ('>',v2,'<=',v1)), I(('>',v1))


	if (c1=='>=') and (c2=='<'):
		if (v1 < v2):
			return I(('>=',v2)), I(('>=',v1,'<',v2)), I(('<',v1))
		if (v1 == v2):
			return I(('>=',v1)), [], I(('<',v1))
		if (v1 > v2):
			return I(('>=',v1)), [], I(('<',v2))

	if (c1=='>=') and (c2=='>'):
		if (v1 < v2):
			return I(('>=',v1,'<=',v2)), I(('>',v2)), []
		if (v1 == v2):
			return I(('=',v1)), I(('>',v1)), []
		if (v1 > v2):
			return [], I(('>=',v1)), I(('>',v2,'<',v1))

	if (c1=='>=') and (c2=='<='):
		if (v1 < v2):
			return I(('>',v2)), I(('>=',v1,'<=',v2)), I(('<',v1))
		if (v1 == v2):
			return I(('>',v1)), I(('=',v1)), I(('<',v1))
		if (v1 > v2):
			return I(('>=',v1)), [], I(('<=',v2))

	if (c1=='>=') and (c2=='>='):
		if (v1 < v2):
			return I(('>=',v1,'<',v2)), I(('>=',v2)), []
		if (v1 == v2):
			return [], I(('>=',v1)), []
		if (v1 > v2):
			return [], I(('>=',v1)), I(('>=',v2,'<',v1))

	if (c1=='>=') and (c2=='='):
		if (v1 < v2):
			return I(('>=',v1,'<',v2),('>',v2)), I(('=',v2)), []
		if (v1 == v2):
			return I(('>',v1)), I(('=',v1)), []
		if (v1 > v2):
			return I(('>=',v1)), [], I(('=',v2))

	if (c1=='>=') and (c2=='!='):
		if (v1 < v2):
			return I(('=',v2)), I(('>=',v1,'<',v2),('>',v2)), I(('<',v1))
		if (v1 == v2):
			return I(('=',v1)), I(('>',v1)), I(('<',v1))
		if (v1 > v2):
			return [], I(('>=',v1)), I(('<',v2),('>',v2,'<',v1))



	if (c1=='=') and (c2=='<'):
		if (v1 < v2):
			return [], I(('=',v1)), I(('<',v1),('>',v1,'<',v2))
		if (v1 == v2):
			return I(('=',v1)), [], I(('<',v1))
		if (v1 > v2):
			return I(('=',v1)), [], I(('<',v2))

	if (c1=='=') and (c2=='>'):
		if (v1 < v2):
			return I(('=',v1)), [], I(('>',v2))
		if (v1 == v2):
			return I(('=',v1)), [], I(('>',v1))
		if (v1 > v2):
			return [], I(('=',v1)), I(('>',v2,'<',v1),('>',v1))

	if (c1=='=') and (c2=='<='):
		if (v1 < v2):
			return [], I(('=',v1)), I(('<',v1),('>',v1,'<=',v2))
		if (v1 == v2):
			return [], I(('=',v1)), I(('<',v1))
		if (v1 > v2):
			return I(('=',v1)), [], I(('<=',v2))

	if (c1=='=') and (c2=='>='):
		if (v1 < v2):
			return I(('=',v1)), [], I(('>=',v2))
		if (v1 == v2):
			return [], I(('=',v1)), I(('>',v2))
		if (v1 > v2):
			return [], I(('=',v1)), I(('>=',v2,'<',v1),('>',v1))

	if (c1=='=') and (c2=='='):
		if (v1 < v2):
			return I(('=',v1)), [], I(('=',v2))
		if (v1 == v2):
			return [], I(('=',v1)), []
		if (v1 > v2):
			return I(('=',v1)), [], I(('=',v2))

	if (c1=='=') and (c2=='!='):
		if (v1 < v2):
			return [], I(('=',v1)), I(('<',v1),('>',v1,'<',v2),('>',v2))
		if (v1 == v2):
			return I(('=',v2)), [], I(('<',v1),('>',v1))
		if (v1 > v2):
			return [], I(('=',v1)), I(('<',v2),('>',v2,'<',v1),('>',v1))



	if (c1=='!=') and (c2=='<'):
		if (v1 < v2):
			return I(('>=',v2)), I(('<',v1),('>',v1,'<',v2)), I(('=',v1))
		if (v1 == v2):
			return I(('>',v1)), I(('<',v1)), []
		if (v1 > v2):
			return I(('>=',v2,'<',v1),('>',v1)), I(('<',v2)), []

	if (c1=='!=') and (c2=='>'):
		if (v1 < v2):
			return I(('<',v1),('>',v1,'<=',v2)), I(('>',v2)), []
		if (v1 == v2):
			return I(('<',v1)), I(('>',v1)), []
		if (v1 > v2):
			return I(('<=',v2)), I(('>',v2,'<',v1),('>',v1)), I(('=',v1))

	if (c1=='!=') and (c2=='<='):
		if (v1 < v2):
			return I(('>',v2)), I(('<',v1),('>',v1,'<=',v2)), I(('=',v1))
		if (v1 == v2):
			return I(('>',v1)), I(('<',v1)), I(('=',v1))
		if (v1 > v2):
			return I(('>',v2,'<',v1),('>',v1)), I(('<=',v2)), []

	if (c1=='!=') and (c2=='>='):
		if (v1 < v2):
			return I(('<',v1),('>',v1,'<',v2)), I(('>=',v2)), []
		if (v1 == v2):
			return I(('<',v1)), I(('>',v1)), I(('=',v1))
		if (v1 > v2):
			return I(('<',v2)), I(('>=',v2,'<',v1),('>',v1)), I(('=',v1))

	if (c1=='!=') and (c2=='='):
		if (v1 < v2):
			return I(('<',v1),('>',v1,'<',v2),('>',v2)),  I(('=',v2)), []
		if (v1 == v2):
			return I(('<',v1),('>',v1)), [], I(('=',v1))
		if (v1 > v2):
			return I(('<',v2),('>',v2,'<',v1),('>',v1)), I(('=',v2)), []

	if (c1=='!=') and (c2=='!='):
		if (v1 < v2):
			return I(('=',v2)), I(('<',v1),('>',v1,'<',v2),('>',v2)), I(('=',v1))
		if (v1 == v2):
			return [], I(('<',v1),('>',v1)), []
		if (v1 > v2):
			return I(('=',v2)), I(('<',v2),('>',v2,'<',v1),('>',v1)), I(('=',v1))


def negate_DNF(E):
	and_terms = []
	for and_term in E._terms:
		if len(and_terms) == 0:
			and_terms = [ expressions.AndExpression([ RVFuncs.logical_not(t) ]) for t in and_term._terms ]
		else:
			and_terms = [ expressions.AndExpression([ *t0._terms, RVFuncs.logical_not(t1) ]) for t0 in and_terms for t1 in and_term._terms ]
	return expressions.OrExpression(and_terms)

def comparator_implications(c1, v1, c2, v2):
	# returns true if comparison 1 implies comparison 2
	if c1 == '>' and c2 in ['>', '>=', '!=']:
		return (v1 >= v2)
	if c1 == '>=' and c2 in ['>', '!=']:
		return (v1 > v2)
	if c1 == '>=' and c2 == '>=':
		return (v1 >= v2)


	if c1 == '<' and c2 in ['<', '<=', '!=']:
		return (v1 <= v2)
	if c1 == '<=' and c2 in ['<', '!=']:
		return (v1 < v2)
	if c1 == '<=' and c2 == '<=':
		return (v1 <= v2)

	if c1 == '=' and c2 == '>':
		return (v1 > v2)
	if c1 == '=' and c2 == '<':
		return (v1 < v2)
	if c1 == '=' and c2 == '>=':
		return (v1 >= v2)
	if c1 == '=' and c2 == '<=':
		return (v1 <= v2)
	if c1 == '=' and c2 == '=':
		return (v1 == v2)
	if c1 == '=' and c2 == '!=':
		return (v1 != v2)

	if c1 == '!=' and c2 == '!=':
		return (v1 == v2)
	return False

def partition_comparisons(C1, C2):
	n1, c1, v1 = C1.variable, C1.comparator, C1.value.value
	n2, c2, v2 = C2.variable, C2.comparator, C2.value.value
	assert n1 == n2, 'C1 and C2 do not refer to the same variables.'
	if c1 == '>':
		if c2 == '>':
			i1 = interval(n1, c1, v1, COMPARATOR_NEGATIONS[c2], v2) if (v2 > v1) else None
			i2 = RVFuncs.comparator_variable(n1, '>', max(v1,v2))
			i3 = interval(n1, c1, v2, COMPARATOR_NEGATIONS[c1], v1) if (v2 < v1) else None
			return i1, i2, i3
		if c2 == '<':
			i1 = RVFuncs.comparator_variable(n1, '>=', v2) if (v2 > v1) else RVFuncs.comparator_variable(n1, '>', v)
			i2 = interval(n1, )


			interval(n1, c1, v1, COMPARATOR_NEGATIONS[c2], v2) if (v2 > v1) else None
			i2 = RVFuncs.comparator_variable(n1, '>', max(v1,v2))
			i3 = interval(n1, c1, v2, COMPARATOR_NEGATIONS[c1], v1) if (v2 < v1) else None
			return i1, i2, i3


# def simplify_and_expression(E):
# 	terms = []
# 	for t0 in E._terms:
# 		if len(terms) == 0:
# 			terms.append(t0)
# 		else:
# 			shared_terms = [ t1 for t1 in terms if t1.variable == t0.variable ]
# 			if len(shared_terms) == 0:
# 				terms.append(t0)
# 			else:
# 				new_shared_terms = []
# 				for t1 in shared_terms:
# 					if t1.comparator == ''
# 			print(shared_terms)
# def partition():
