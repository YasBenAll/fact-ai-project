import numpy as np
from utils.rvs import ConstraintManager

if __name__ == '__main__':

	c1 = "|E[a]*E[c]/(E[d]/E[d])+E[e]|-1"
	c2 = "E[e] + E[f]-1"

	trivial_bounds = {'a' : (0,1),
					  'c' : (0,1) }

	C = ConstraintManager([c1, c2], trivial_bounds=trivial_bounds)
	
	def test(exp, pad=100):
		print('%s    -->   %r' % (('%r'%exp).rjust(pad), C.parse(exp)))

	test("-E[x]*E[y]")
	test("E[x]*-E[y]")
	test("-E[x]*-E[y]")
	test("-E[x]*-E[y]*-E[z]/E[q]/E[r]")
	test("-E[x]*-E[y]/-E[q]/-E[r]")
	test("-E[x]/-E[q]*-E[y]/-E[r]")
	test("-E[x]*-E[y]*-E[z]/-(E[q]*E[r])")
	test("|E[a]*E[c]/(E[d]/E[d])+E[e]|-1")
	test("|E[d]*E[a]*E[c]/(E[d]/E[d])+E[e]|-1")
	test("max(E[a],E[b])/max(E[b],E[c])")
	test("max(E[a],E[b])/max(E[b],E[a])")
	test("max(E[a],E[b])/min(E[b],E[a])")
	test("max(E[a],E[b])/min(-E[b],-E[a])")
	test("max(E[a],E[b],E[c])")
	test("max(E[a],max(E[b],E[c]))")
	test("max(E[a],min(E[b],E[c]))")
	test("min(E[a],min(E[b],E[c]))")
	test("E[x]/E[y] + E[z]*E[x]/E[y] - E[x]/E[y] - E[x]*E[z]*E[y]/(E[y]*E[y])")
	test("E[x]*E[x]/E[y] + E[z]*E[x]*E[r]/E[y] - E[x]/E[y] - E[z]*E[x]*E[z]*E[y]/(E[y]*E[y])")
	test("max(-E[x]*E[x]/3, -3*E[x]*E[z]/(E[x]*E[x]*E[z]))")
	test("max(-E[x]*E[x]/3/E[x], -3*E[x]*E[z]/(E[x]*E[x]*E[z]))")
	test("|E[a]-E[b]|+|E[b]-E[c]|/|E[d]-E[e]|*max(E[d]-E[e],-E[d]+E[e])+2")