import numpy as np
from utils.rvs.parser import get_parser
from utils.rvs import logic 
from utils.rvs.logic import partition, partition_intervals, process_interval
from utils.rvs.expressions import RVFuncs
from matplotlib import pyplot as plt
from utils.rvs import expressions

def test(s):
	p = get_parser(mode='event')
	E = p.parse(s)
	print(E, '  -->  ', logic.convert_to_DNF(E))




def plot_comparator(ax, E, vmin, vmax, y, color='r'):
	if isinstance(E, expressions.ComparatorExpression):
		c, v = E.comparator, E.value.value
		if c in ['<=','<','!=']:
			ax.plot([vmin, v], [y,y], marker=None, color=color, zorder=-1)
			ax.plot([vmin], [y], marker=8, color=color)
		if c in ['>=','>','!=']:
			ax.plot([v, vmax], [y,y], marker=None, color=color, zorder=-1)
			ax.plot([vmax], [y], marker=9, color=color)
		ax.scatter([v],[y], marker='o', c=(color if c in ['>=', '<=', '='] else 'white'), edgecolors=color, zorder=1)
	elif isinstance(E, expressions.AndExpression):
		E1, E2 = E._terms
		c1, v1 = E1.comparator, E1.value.value
		c2, v2 = E2.comparator, E2.value.value
		if (v1 < v2):
			assert '>' in c1 and '<' in c2, ('Invalid interval: %r' % E)
		elif (v2 < v1):
			assert '>' in c2 and '<' in c1, ('Invalid interval: %r' % E)
		else:
			assert '=' in c1 and '=' in c2, ('Invalid interval: %r' % E)
		ax.plot([min(v1,v2), max(v1,v2)], [y,y], marker=None, color=color, zorder=-1)
		ax.scatter([v1],[y], marker='o', c=(color if c1 in ['>=', '<=', '='] else 'white'), edgecolors=color, zorder=1)
		ax.scatter([v2],[y], marker='o', c=(color if c2 in ['>=', '<=', '='] else 'white'), edgecolors=color, zorder=1)


def gen_subplot(ax, E1, E2, vmin=-1, vmax=1):
	plot_comparator(ax, E1, vmin, vmax, 4.5, color='r')
	plot_comparator(ax, E2, vmin, vmax, 4, color='b')

	ax.axhline(3, vmin, vmax, color='k', ls=':')

	A_only, AB, B_only = logic.partition(E1,E2)

	try:
		for E in A_only:
			plot_comparator(ax, E, vmin, vmax, 2, color='r')
		for E in AB:
			plot_comparator(ax, E, vmin, vmax, 1.5, color='g')
		for E in B_only:
			plot_comparator(ax, E, vmin, vmax, 1, color='b')
	except AssertionError as e:
		print("Exception:")
		print('E1:',E1,' E2:',E2)
		raise(e)

	# ax.axis('off')
	ax.xaxis.set_ticks([])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticks([])
	ax.yaxis.set_ticklabels([])
	ax.set_ylim((0.5,5))
	ax.set_title('%r,  %r' % (E1,E2))

	return A_only, AB, B_only



def gen_plot(c1, v1, v2):
	v1, v2 = min(v1,v2), max(v1,v2)
	v3 = (v1+v2)/2
	margin = 1.1*(v2 - v1)
	vmin = v1 - margin
	vmax = v2 + margin

	comparisons = ['<','>','<=','>=','=','!=']
	p = get_parser(mode='event')
	E = lambda c, v: p.parse('X %s %r' % (c,v))
	fig, axes = plt.subplots(6,3)
	for ax_row, c2 in zip(axes, comparisons):
		gen_subplot(ax_row[0], E(c1,v1), E(c2,v2), vmin=vmin, vmax=vmax)
		gen_subplot(ax_row[1], E(c1,v3), E(c2,v3), vmin=vmin, vmax=vmax)
		gen_subplot(ax_row[2], E(c1,v2), E(c2,v1), vmin=vmin, vmax=vmax)
	fig.tight_layout()
	return fig


def gen_full_plot(c1, v1, v2):
	v1, v2 = min(v1,v2), max(v1,v2)
	v3 = (v1+v2)/2
	margin = 1.1*(v2 - v1)
	vmin = v1 - margin
	vmax = v2 + margin

	p = get_parser(mode='event')
	E = lambda c, v: p.parse('X %s %r' % (c,v))
	fig, axes = plt.subplots(12,9)
	comparisons = ['<','>','<=','>=','=','!=']
	_comparisons = [['<','>','='],['<=','>=','!=']]
	for _i in range(len(comparisons)):
		i = _i // 3
		j = (_i-i*3)
		c1 = _comparisons[i][j]
		for _j,c2 in enumerate(comparisons):
			r = i*6+_j
			c = j*3
			gen_subplot(axes[r,c+0], E(c1,v1), E(c2,v2), vmin=vmin, vmax=vmax)
			gen_subplot(axes[r,c+1], E(c1,v3), E(c2,v3), vmin=vmin, vmax=vmax)
			gen_subplot(axes[r,c+2], E(c1,v2), E(c2,v1), vmin=vmin, vmax=vmax)

	fig.subplots_adjust(bottom=0.03,top=0.96, left=0.03, right=0.97,hspace=.6)
	return fig




if __name__ == '__main__':

	p = get_parser(mode='event')
	test('A=1,(B=2||C=3)')
	test('A=1,(B=2||C=3),(D=4||E=5)')
	test('A=1,(B=2||C=3,D=4)')
	test('A=1||(B=2||C=3),(D=4||E=5)')

	E = p.parse('A=1,(B=2||C=3),(D=4||E=5)')
	E = logic.convert_to_DNF(E)
	print(logic.negate_DNF(E))

	def test2(s1, s2):
		E1 = p.parse(s1)
		E2 = p.parse(s2)
		a, b, c = logic.partition_comparisons(E1, E2)
		print()
		print(a)
		print(b)
		print(c)

	test2('X > 1', 'X > 2')
	test2('X > 2', 'X > 1')

	# gen_plot('<',-1,1).show()
	# gen_plot('>',-1,1).show()
	# gen_plot('<=',-1,1).show()
	# gen_plot('>=',-1,1).show()
	# gen_plot('=',-1,1).show()
	# gen_plot('!=',-1,1).show()

	E1 = p.parse('X>=1, X<=1')
	E2 = p.parse('X>0,X<1.5')
	E3 = p.parse('X>0.5,X<=1')
	E4 = p.parse('X>=2,X<=2')
	print('A')
	print('OUTPUT:', partition_intervals(E1,E2))
	print('B')
	print('OUTPUT:', partition_intervals(partition_intervals(E1,E2)[1][0], E3))
	print('C')
	print('OUTPUT:', partition_intervals(p.parse('X>0,X<1'),p.parse('X>1,X<2')))
	print('D')
	print('OUTPUT:', partition_intervals(partition_intervals(E1,E2)[1][0], E4))