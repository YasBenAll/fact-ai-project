from parglare import Parser, Grammar
from utils.rvs.expressions import RVFuncs


### Production rules for outer expressions, inner expressions, and events

expression_terminals = r"""
	variable: /[#]?[a-zA-Z_$][a-zA-Z_$0-9]*/;
	number: /\d+(\.\d+)?/;
"""

expression_productions = r"""
	expr
	: term
	| expr '+' term
	| expr '-' term
	;

	exprs
	: expr
	| exprs ',' expr
	;

	term
	: unary
	| term '*' unary
	| term '/' unary
	;

	unary
	: primary
	| '|' expr '|'
	| '+' unary
	| '-' unary
	| 'max(' exprs ')'
	| 'min(' exprs ')'
	| 'nanmax(' exprs ')'
	| 'nanmin(' exprs ')'
	;

	primary
	: expected_value
	| number
	| '(' expr ')'
	;


	expected_value
	: 'E{' variable '(' inner_expr ')}[' sample_set ']'
	| 'E[' sample_set ']'
	;

	sample_set
	: inner_expr 
	| inner_expr '|' event_expr
	| event_expr 
	| event_expr '|' event_expr
	;
"""

inner_expression_productions = r"""
	inner_expr
	: inner_term
	| inner_expr '+' inner_term
	| inner_expr '-' inner_term
	;

	inner_term
	: inner_unary
	| inner_term '*' inner_unary
	| inner_term '/' inner_unary
	;

	inner_unary
	: inner_primary
	| '|' inner_expr '|'
	| '+' inner_unary
	| '-' inner_unary
	;

	inner_primary
	: number
	| variable
	| '(' inner_expr ')'
	;
"""

event_productions = r"""
	event_expr
	: event_term
	| event_expr '||' event_term
	;

	event_term
	: event_unary
	| event_unary ',' event_term
	;

	event_unary
	: comparison
	| '~' event_unary
	| '(' event_expr ')'
	;

	comparison
	: inner_expr inequality inner_expr
	| inner_expr equality inner_expr
	| inner_expr inequality equality inner_expr
	;
"""

event_terminals = r"""
	inequality: /[<|>|!]/;
	equality: /[=]/;
"""

### Actions for outer expressions, inner expressions, and events

expression_production_actions = {
	"expr":   [  lambda _, nodes: nodes[0],
			     lambda _, nodes: RVFuncs.sum(nodes[0], nodes[2]),
			     lambda _, nodes: RVFuncs.sum(nodes[0], RVFuncs.negative(nodes[2]))],
	"exprs":  [  lambda _, nodes: nodes[0],
			     lambda _, nodes: ([*nodes[0], nodes[2]] if isinstance(nodes[0], list) else [nodes[0],nodes[2]])],
	"term":   [  lambda _, nodes: nodes[0],
			     lambda _, nodes: RVFuncs.product(nodes[0], nodes[2]),
			     lambda _, nodes: RVFuncs.fraction(nodes[0], nodes[2])],
    "unary":  [  lambda _, nodes: nodes[0],
			     lambda _, nodes: RVFuncs.abs(nodes[1]),
    			 lambda _, nodes: nodes[1],
    			 lambda _, nodes: RVFuncs.negative(nodes[1]),
			     lambda _, nodes: RVFuncs.max(*nodes[1]), 
			     lambda _, nodes: RVFuncs.min(*nodes[1]),
			     lambda _, nodes: RVFuncs.nanmax(*nodes[1]), 
			     lambda _, nodes: RVFuncs.nanmin(*nodes[1])], 
    "primary": [ lambda _, nodes: nodes[0],
    			 lambda _, nodes: RVFuncs.constant(nodes[0]),
    			 lambda _, nodes: nodes[1]],	
    "expected_value": [
    			 lambda _, nodes: RVFuncs.expected_value(nodes[5], is_func=nodes[1], is_expr=nodes[3]),
    			 lambda _, nodes: RVFuncs.expected_value(nodes[1])],
    "sample_set": [
    			 lambda _, nodes: RVFuncs.sample_set(nodes[0]),
    			 lambda _, nodes: RVFuncs.sample_set(nodes[0],nodes[2]),
    			 lambda _, nodes: RVFuncs.sample_set(nodes[0]),
    			 lambda _, nodes: RVFuncs.sample_set(nodes[0],nodes[2])],
}

expression_terminal_actions = {	
    "number":    lambda _, value: value,
    "variable":  lambda _, value: value,
}

inner_expression_production_actions = {
	"inner_expr":   [ 
				 lambda _, nodes: nodes[0],
			     lambda _, nodes: RVFuncs.sum(nodes[0], nodes[2]),
			     lambda _, nodes: RVFuncs.sum(nodes[0], RVFuncs.negative(nodes[2]))],
	"inner_term":   [ 
				 lambda _, nodes: nodes[0],
			     lambda _, nodes: RVFuncs.product(nodes[0], nodes[2]),
			     lambda _, nodes: RVFuncs.fraction(nodes[0], nodes[2])],
    "inner_unary":  [ 
    			 lambda _, nodes: nodes[0],
			     lambda _, nodes: RVFuncs.abs(nodes[1]),
    			 lambda _, nodes: nodes[1],
    			 lambda _, nodes: RVFuncs.negative(nodes[1])], 
    "inner_primary": [ 
    			 lambda _, nodes: RVFuncs.constant(nodes[0]),
    			 lambda _, nodes: RVFuncs.variable(nodes[0]),
    			 lambda _, nodes: nodes[1]],
}

event_production_actions = {
    "event_expr": [
    			 lambda _, nodes: nodes[0],
    			 lambda _, nodes: RVFuncs.logical_or([ nodes[0], nodes[2] ])],
    "event_term": [
    			 lambda _, nodes: nodes[0],
    			 lambda _, nodes: RVFuncs.logical_and([ nodes[0], nodes[2] ])],
    "event_unary": [
    			 lambda _, nodes: nodes[0],
    			 lambda _, nodes: RVFuncs.logical_not(nodes[1]),
    			 lambda _, nodes: nodes[1]],
    "comparison": [
    			 lambda _, nodes: RVFuncs.comparator_variable(nodes[0], nodes[1], nodes[2]),
    			 lambda _, nodes: RVFuncs.comparator_variable(nodes[0], nodes[1], nodes[2]),
    			 lambda _, nodes: RVFuncs.comparator_variable(nodes[0], nodes[1]+nodes[2], nodes[3])]
}

event_terminal_actions = {
    "inequality": lambda _, value: value,
    "equality":   lambda _, value: value
}



### Final grammar and action specifications
expression_grammar = '%s\n%s\n%s\nterminals\n%s\n%s' % (expression_productions, inner_expression_productions, event_productions, expression_terminals, event_terminals)
expression_actions = dict( **expression_production_actions,
						   **inner_expression_production_actions,
						   **event_production_actions, 
						   **expression_terminal_actions,
						   **event_terminal_actions)
inner_expression_grammar = '%s\n%s\nterminals\n%s\n%s' % (inner_expression_productions, event_productions, expression_terminals, event_terminals)
inner_expression_actions = dict(**inner_expression_production_actions,
								**event_production_actions, 
								**expression_terminal_actions,
								**event_terminal_actions)
event_grammar = '%s\n%s\nterminals\n%s\n%s' % (event_productions, inner_expression_productions, expression_terminals, event_terminals)
event_actions = dict(**inner_expression_production_actions,
					 **event_production_actions, 
					 **expression_terminal_actions,
					 **event_terminal_actions)

grammars = {
	'outer' : expression_grammar,
	'inner' : inner_expression_grammar,
	'event' : event_grammar
}

action_sets = {
	'outer' : expression_actions,
	'inner' : inner_expression_actions,
	'event' : event_actions
}


def get_parser(debug=False, mode='outer'):
	g = Grammar.from_string(grammars[mode.lower()])
	return Parser(g, debug=debug, actions=action_sets[mode.lower()])
