import numpy as np
from copy import copy, deepcopy
from utils.rvs.utils import COMPARATOR_NEGATIONS

def get_constant_name(counter={'c':0}):
	name = 'c%d' % counter['c']
	counter['c'] += 1
	return name

def get_variable_name(counter={'v':0}):
	name = 'v%d' % counter['v']
	counter['v'] += 1
	return name

def get_expression_name(counter={'e':0}):
	name = 'e%d' % counter['c']
	counter['e'] += 1
	return name





class Expression():
	def __init__(self):
		self.trivial_bounds = None
		self._terms = []
	def __eq__(self, E):
		return isinstance(E, self.__class__) and all([ T==_T for (T,_T) in zip(self._terms,E._terms)])

class CommutativeExpression(Expression):
	def __init__(self):
		super().__init__()
	def __eq__(self,E):
		if not(isinstance(E, self.__class__)):
			return False
		terms, _terms = copy(self._terms), copy(E._terms)
		try:
			for term in terms:
				_terms.remove(term)
		except ValueError:
			return False
		return len(_terms) == 0

class NoncommutativeExpression(Expression):
	def __init__(self):
		super().__init__()
	def __eq__(self,E):
		return isinstance(E, self.__class__) and all([ T==_T for (T,_T) in zip(self._terms,E._terms) ])

class SingleTermExpression():
	pass


class SampleSet(Expression):
	def __init__(self, expression, condition=None):
		super().__init__()
		self.expression = expression
		self.condition  = condition


class ConstantExpression(Expression, SingleTermExpression):
	def __init__(self, name, value):
		super().__init__()
		self.name = get_constant_name()
		self.value = value
	def __repr__(self):
		return  str(self.value)
	def __eq__(self, E):
		return isinstance(E,self.__class__) and self.value == E.value

class VariableExpression(Expression, SingleTermExpression):
	def __init__(self, name):
		super().__init__()
		if name.startswith('#'):
			self.name = name[1:]
			self._special = 'index'
		else:
			self.name = name
			self._special = None
	def __repr__(self):
		return self.name
	def __eq__(self, E):
		return isinstance(E,self.__class__) and self.name == E.name and self._special == E._special

class SampleSet(Expression, SingleTermExpression):
	def __init__(self, expression, condition=None):
		super().__init__()
		name = '%r' % expression
		if not(condition is None):
			name += '|%r' % condition
		self.name = '[%s]' % name
		self.expression = expression
		self.condition  = condition
	def __repr__(self):
		return self.name
	def __eq__(self, E):
		return isinstance(E,self.__class__) and (self.expression == E.expression) and (self.condition == E.condition)


class ExpectedValue(Expression, SingleTermExpression):
	def __init__(self, sample_set, is_func=None, is_expr=None):
		super().__init__()
		if is_func is None:
			self.name = 'E%s' % sample_set.name
		else:
			self.name = 'E{%s(%s)}%s' % (is_func, is_expr.name, sample_set.name)
		# self.name = 'E{%s}%s' % () + sample_set.name
		self.sample_set = sample_set
		self._is_func = is_func
		self._is_expr = is_expr
	def __repr__(self):
		return self.name
	def __eq__(self, E):
		if not(isinstance(E,self.__class__)):
			return False
		if not(self.sample_set == E.sample_set):
			return False
		if self._is_func is None and E._is_func is None:
			return True
		else:
			return (self._is_func == E._is_func) and (self._is_expr == E._is_expr)




class ComparatorExpression(VariableExpression):
	def __init__(self, term1, comp, term2):
		name = '%r %s %r' % (term1, comp, term2)
		super().__init__(name)
		self.variable = term1
		self.comparator = comp
		self.value = term2


class NegativeExpression(NoncommutativeExpression, SingleTermExpression):
	def __init__(self, expression):
		super().__init__()
		self._terms = [expression]
	def __repr__(self):
		if isinstance(self._terms[0], SumExpression):
			return '-(%r)' % self._terms[0]	
		return '-%r' % self._terms[0]
	def __eq__(self, E):
		if isinstance(E,self.__class__) and (self._terms[0]==E._terms[0]):
			return True
		if isinstance(E, SumExpression):
			return E == self
		return False


class NotExpression(NoncommutativeExpression, SingleTermExpression):
	def __init__(self, expression):
		super().__init__()
		self._terms = [expression]
	def __repr__(self):
		return '~(%r)' % self._terms[0]	
	def __eq__(self, E):
		if isinstance(E,self.__class__) and (self._terms[0]==E._terms[0]):
			return True
		return False


class AbsExpression(NoncommutativeExpression, SingleTermExpression):
	def __init__(self, expression):
		super().__init__()
		self._terms = [expression]
	def __repr__(self):
		return '|%r|' % self._terms[0]	
	def __eq__(self, E):
		return isinstance(E,self.__class__) and (self._terms[0]==E._terms[0]) 

class FractionExpression(NoncommutativeExpression):
	def __init__(self, num, den):
		super().__init__()
		self._terms = [num, den]
	def __repr__(self):
		num, den = self._terms
		num_str = '(%r)'%num if isinstance(num, SumExpression) else '%r'%num
		den_str = '%r'%den if isinstance(den, SingleTermExpression) else '(%r)'%den
		return '%s/%s' % (num_str, den_str)
	def __eq__(self, E):
		return isinstance(E, self.__class__) and (self._terms[0]==E._terms[0]) and (self._terms[1]==E._terms[1])

class SumExpression(CommutativeExpression):
	def __init__(self, expressions):
		super().__init__()
		self._terms = list(expressions)
	def __repr__(self):
		string = '%r' % self._terms[0]
		for t in self._terms[1:]:
			string += '%r'%t if isinstance(t, NegativeExpression) else '+%r'%t
		return string
	def __eq__(self, E):
		if super().__eq__(E):
			return True
		if isinstance(E, NegativeExpression):
			return E == RVFuncs.negative(SumExpression([ RVFuncs.negative(e) for e in self._terms ]))
		return False

class AndExpression(CommutativeExpression):
	def __init__(self, comparisons):
		super().__init__()
		self._terms = list(comparisons)
		self.name = ','.join('%s'%c.name for c in comparisons)
	def __repr__(self):
		return ','.join([('(%r)' % t) if isinstance(t,OrExpression) else ('%r' % t) for t in self._terms])
	def __eq__(self, E):
		return super().__eq__(E)


class OrExpression(CommutativeExpression):
	def __init__(self, comparisons):
		super().__init__()
		self._terms = list(comparisons)
		self.name = '||'.join('%s'%c.name for c in comparisons)
	def __repr__(self):
		return '||'.join([('(%r)' % t) if isinstance(t,AndExpression) else ('%r' % t) for t in self._terms])
	def __eq__(self, E):
		return super().__eq__(E)

class ProductExpression(CommutativeExpression):
	def __init__(self, expressions):
		super().__init__()
		self._terms = list(expressions)
	def __repr__(self):
		string = '(%r)'%self._terms[0] if (isinstance(self._terms[0], SumExpression) and len(self._terms) > 1) else '%r'%self._terms[0]
		for t in self._terms[1:]:
			string += '*(%r)'%t if isinstance(t, SumExpression) else '*%r'%t
		return string
	
class MaxExpression(CommutativeExpression, SingleTermExpression):
	def __init__(self, expressions):
		super().__init__()
		self._terms = list(expressions)
	def __repr__(self):
		return 'MAX{%s}' % ', '.join([ '%r'%t for t in self._terms ])

class MaxRecipExpression(Expression, SingleTermExpression):
	def __init__(self, expression):
		super().__init__()
		self._terms = [expression]
	def __repr__(self):
		return 'MAX{%s, %s}' % (self._terms[0], RVFuncs.fraction(RVFuncs.constant(1), self._terms[0]))

class NANMaxExpression(CommutativeExpression, SingleTermExpression):
	def __init__(self, expressions):
		super().__init__()
		self._terms = list(expressions)
	def __repr__(self):
		return 'NANMAX{%s}' % ', '.join([ '%r'%t for t in self._terms ])













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

def parse_value(value):
	''' Attempts to interpret <value> as a number. '''
	if isinstance(value, str):
		try:
			value = int(value)
		except ValueError:
			value = float(value)
	return value

class RVFuncs():
	@staticmethod
	def constant(value_raw):
		value = parse_value(value_raw)
		return ConstantExpression('c', value)

	@staticmethod
	def variable(name):
		return VariableExpression(name)

	@staticmethod
	def comparator_variable(term1, comp, term2):
		return ComparatorExpression(term1, comp, term2)

	@staticmethod
	def sample_set(variable, condition=None):
		return SampleSet(variable, condition)

	@staticmethod
	def expected_value(sampleset, is_func=None, is_expr=None):
		return ExpectedValue(sampleset, is_func=is_func, is_expr=is_expr)

	@staticmethod
	def negative(e):
		''' Returns the negative of <e>, reducing nested negatives. '''
		n_negatives = 1
		while isinstance(e, NegativeExpression):
			e = e._terms[0]
			n_negatives += 1
		if isinstance(e, ConstantExpression):
			return RVFuncs.constant(-e.value if (n_negatives % 2 == 1) else e.value)
		return NegativeExpression(e) if (n_negatives % 2 == 1) else e

	@staticmethod
	def logical_not(e):
		n_nots = 1
		while isinstance(e, NotExpression):
			e = e._terms[0]
			n_nots += 1
		if (n_nots % 2 == 0):
			return e
		if isinstance(e, ComparatorExpression):
			return ComparatorExpression(e.variable, COMPARATOR_NEGATIONS[e.comparator], e.value)
		return NotExpression(e)

	@staticmethod
	def sum(*expressions):
		''' Returns the sum of <expressions>, factoring out constants and shared factors. '''
		# Aggregate terms that are sums themselves
		exps = []
		for e in expressions:
			if isinstance(e, SumExpression):
				exps.extend(e._terms)
			else:
				exps.append(e)
		expressions = exps
				
		# Aggregate terms that are constants
		cval = 0
		exps = []
		for e in expressions:
			if isinstance(e, ConstantExpression):
				cval += e.value
			elif isinstance(e, NegativeExpression) and isinstance(e._terms[0], ConstantExpression):
				cval -= e._terms[0].value
			else:
				exps.append(e)
		if cval != 0 or len(exps) == 0:
			const = RVFuncs.constant(cval)
			exps = [ const, *exps]
		expressions = exps

		if len(expressions) == 1:
			return expressions[0]




		# Check if all terms share a common denominator and factor it out
		def split_as_fraction(e):
			if isinstance(e, FractionExpression):
				return [e._terms[0], e._terms[1]]
			elif isinstance(e, NegativeExpression) and isinstance(e._terms[0],FractionExpression):
				return [RVFuncs.negative(e._terms[0]._terms[0]), e._terms[0]._terms[1]]
			return [e, None]
		nums, dens = zip(*[ split_as_fraction(e) for e in exps ])
		if all([ not(dens[0] is None) and d==dens[0] for d in dens ]):
			exps = nums
			common_den = dens[0]
		else:
			common_den = None

		# Check if any terms have shared product factors and factor them out
		def extract_unsigned_terms(e):
			if isinstance(e, NegativeExpression) or isinstance(e, FractionExpression):
				return extract_unsigned_terms(e._terms[0])
			if isinstance(e, ProductExpression):
				return e._terms
			return [e]
		def remove_terms(e, terms):
			if isinstance(e, NegativeExpression):
				return RVFuncs.negative(remove_terms(e._terms[0], terms))
			if isinstance(e, FractionExpression):
				return RVFuncs.fraction(remove_terms(e._terms[0], terms), e._terms[1])
			if isinstance(e, ProductExpression):
				remaining = e._terms.copy()
				for t in terms:
					remaining.remove(t)
				return RVFuncs.product(*remaining) if len(remaining) > 0 else RVFuncs.constant(1)
			return RVFuncs.constant(1) if len(terms) > 0 else e
		has_negative   = [ isinstance(e,NegativeExpression) for e in exps ]
		unsigned_terms = [ extract_unsigned_terms(e)        for e in exps ]
		unsigned_terms_tmp = deepcopy(unsigned_terms)
		shared_terms = []
		for st in unsigned_terms[0]:
			if isinstance(st, ConstantExpression) and (st.value == 1):
				continue
			if all([ (st in terms) for terms in unsigned_terms_tmp[1:] ]):
				shared_terms.append(st)
				for terms in unsigned_terms_tmp:
					terms.remove(st)
		if len(shared_terms) > 0:
			remainder = RVFuncs.sum(*[ remove_terms(e, shared_terms) for e in exps ])
		else:
			remainder = SumExpression(exps)

		# Return the product of the common factor and the remainder sum
		if len(shared_terms) > 0 and common_den is None:
			common_factor = RVFuncs.product(*shared_terms)
			return RVFuncs.product(common_factor, remainder)
		elif len(shared_terms) > 0:
			common_factor = RVFuncs.fraction(RVFuncs.product(*shared_terms), common_den)
			return RVFuncs.product(common_factor, remainder)
		return remainder

	@staticmethod
	def diff(e0, e1):
		return RVFuncs.sum(e0, RVFuncs.negative(e1))

	@staticmethod
	def max(*expressions):
		if len(expressions) == 1:
			return expressions[0]
		exps = []
		for e in expressions:
			if isinstance(e, MaxExpression):
				exps.extend(e._terms)
			else:
				exps.append(e)
		if len(expressions) == 2:
			e1, e2 = expressions
			# If the max *happens* to be Max(E, 1/E) for some E, reduce to a MaxRecip
			if e1 == RVFuncs.fraction(RVFuncs.constant(1), e2):
				return MaxRecipExpression(e1)
			# If the max *happens* to be Max(E, -E) for some E, reduce to Abs
			elif e1 == RVFuncs.negative(e2):
				return AbsExpression(e1)
		return MaxExpression(exps)

	def nanmax(*expressions):
		if len(expressions) == 1:
			return expressions[0]
		exps = []
		for e in expressions:
			if isinstance(e, MaxExpression):
				exps.extend(e._terms)
			else:
				exps.append(e)
		return NANMaxExpression(exps)


	@staticmethod
	def min(*expressions):
		if len(expressions) == 1:
			return expressions[0]
		exps = []
		for e in expressions:
			if isinstance(e, MaxExpression):
				exps.extend(e._terms)
			else:
				exps.append(e)
		# Convert to a negative max
		exps = [ RVFuncs.negative(e) for e in exps ]
		return RVFuncs.negative(RVFuncs.max(*exps))

	@staticmethod
	def abs(e):
		if isinstance(e, NegativeExpression):
			e = e._terms[0]
		return AbsExpression(e)

	@staticmethod
	def pow(e, c):
		return e  # fix

	@staticmethod
	def logical_and(expressions):
		events = []
		for e in expressions:
			if isinstance(e, AndExpression):
				events.extend(e._terms)
			else:
				events.append(e)
		return AndExpression(events)

	@staticmethod
	def logical_or(expressions):
		events = []
		for e in expressions:
			if isinstance(e, OrExpression):
				events.extend(e._terms)
			else:
				events.append(e)
		return OrExpression(events)

	@staticmethod
	def product(*expressions):

		# Strip negatives from input expressions
		n_negatives = 0
		exps = []
		for e in expressions:
			if isinstance(e, NegativeExpression):
				exps.append(e._terms[0])
				n_negatives += 1
			else:
				exps.append(e)
		expressions = exps

		# Remove and input expressions that are a constant 1
		exps = []
		for e in expressions:
			if not(isinstance(e, ConstantExpression) and (e.value == 1)):
				exps.append(e)
		expressions = exps

		# # If there is only one input expression remaining, just return it
		# if len(expressions) == 1:
		# 	return RVFuncs.negative(expressions[0]) if n_negatives % 2 == 1 else expressions[0]

		# If any of the input expressions are a constant equal to 0, return 0
		if any([ isinstance(e,ConstantExpression) and (e.value==0) for e in expressions ]):
			return RVFuncs.constant(0)

		# Aggregate input expressions that are products or fractions
		num_exps = []
		den_exps = []
		for e in expressions:
			if isinstance(e, ProductExpression):
				num_exps.extend(e._terms)
			elif isinstance(e, FractionExpression):
				num_exps.append(e._terms[0])
				den_exps.append(e._terms[1])
			else:
				num_exps.append(e)

		if len(den_exps) > 0:
			# We have a fraction
			num = RVFuncs.product(*num_exps) if len(num_exps) > 1 else num_exps[0]
			den = RVFuncs.product(*den_exps) if len(den_exps) > 1 else den_exps[0]
			expr = RVFuncs.fraction(num, den)
		else:
			# We have a non-fraction product
			# Aggregate constants
			cval = 1
			_exps = []
			for e in num_exps:
				if isinstance(e, ConstantExpression):
					cval = safeprod(cval, e.value)
				else:
					_exps.append(e)
			if len(_exps) == 0:
				expr = RVFuncs.constant(cval)
			elif cval != 1:
				_exps.append(RVFuncs.constant(cval))
				expr = ProductExpression(_exps)
			elif len(_exps) > 1:
				expr = ProductExpression(_exps)
			else:
				expr = _exps[0]

		return expr if (n_negatives % 2 == 0) else RVFuncs.negative(expr)

	@staticmethod
	def fraction(num, den):
		''' Process the numerator and denominator to produce a reduced expression of one of the following forms, in this priority:
		 	 Constant or Variable
			 Negative(Product(PositiveConstant, Fraction)) 
			 Product(PositiveConstant, Fraction)
			 Negative(Fraction).
			Assumes that num and den are already processed into Negative(Product(Constant, Expression)) form. '''

		# Simplify negative signs in the numerator/denominator
		n_negatives = 0
		if isinstance(num, NegativeExpression):
			num = num._terms[0]
			n_negatives += 1
		if isinstance(den, NegativeExpression):
			den = den._terms[0]
			n_negatives += 1

		# Remove any constants in front of the numerator or denominator
		num_val = 1
		den_val = 1
		if isinstance(num, ProductExpression) and isinstance(num._terms[0], ConstantExpression):
			num_val = num._terms[0].value
			num = RVFuncs.product(*num._terms[1:]) if len(num._terms) > 1 else RVFuncs.constant(1)
		if isinstance(den, ProductExpression) and isinstance(den._terms[0], ConstantExpression):
			den_val = den._terms[0].value
			den = RVFuncs.product(*den._terms[1:]) if len(den._terms) > 1 else RVFuncs.constant(1)
		cval = safediv(num_val, den_val)
		if cval < 0:
			n_negatives += 1
			cval = -cval

		# Aggregate terms in the numerator/denominator if one or both are already a fraction
		if isinstance(num, FractionExpression) and isinstance(den, FractionExpression):
			_num = RVFuncs.product(num._terms[0], den._terms[1])
			_den = RVFuncs.product(num._terms[1], den._terms[0])
			num, den = _num, _den
		elif isinstance(num, FractionExpression):	
			_num = num._terms[0]
			_den = RVFuncs.product(num._terms[1], den)
			num, den = _num, _den
		elif isinstance(den, FractionExpression):
			_num = RVFuncs.product(den._terms[1], num)
			_den = den._terms[0]
			num, den = _num, _den

		# Remove terms in products that are present in both the numerator and denominator
		expr = None
		if num == den:
			expr = RVFuncs.constant(1)
		elif isinstance(den, ConstantExpression) and den.value == 1:
			expr = num
		elif isinstance(num, ProductExpression) and isinstance(den, ProductExpression):
			nterms, dterms = copy(num._terms), copy(den._terms)
			for term in nterms:
				if term in den._terms:
					num._terms.remove(term)
					den._terms.remove(term)
			num = RVFuncs.constant(1) if len(num._terms) == 0 else RVFuncs.product(*num._terms)
			den = RVFuncs.constant(1) if len(den._terms) == 0 else RVFuncs.product(*den._terms)
			if isinstance(num, ConstantExpression) and isinstance(den, ConstantExpression):
				expr = RVFuncs.constant(safediv(num.value, den.value))
		elif isinstance(num, ProductExpression) and isinstance(den, SingleTermExpression): 
			if den in num._terms:
				num._terms.remove(den)
				expr = RVFuncs.product(*num._terms)
		elif isinstance(den, ProductExpression) and isinstance(num, SingleTermExpression): 
			if num in den._terms:
				den._terms.remove(num)
				den = RVFuncs.product(*den._terms)
				if isinstance(den, ConstantExpression):
					print(safediv(1,den.value), RVFuncs.constant(safediv(1,den.value)).value)
					expr = RVFuncs.constant(safediv(1,den.value))
				else:
					expr = FractionExpression(RVFuncs.constant(1), RVFuncs.product(*den._terms))
		if expr is None:
			expr = FractionExpression(num, den)


		# Add a constant scaling factor if it is not 1
		if cval != 1:
			constant = RVFuncs.constant(cval)
			expr = RVFuncs.product(constant, expr)
		return RVFuncs.negative(expr) if n_negatives % 2 == 1 else expr


