import csv
import numpy as np

def parse_value(value):
	''' Attempts to interpret <value> as a number. '''
	if isinstance(value, str):
		try:
			value = int(value)
		except ValueError:
			try:
				value = float(value)
			except ValueError:
				pass
	return value

def parse_row(row):
	return [ parse_value(s.strip()) for s in row.split('; ') ]

def parse_data(fname):
	data = []
	for row in open(fname):
		data.append(parse_row(row))
	return data[0], data[1:]

def data_to_dict(header, data):
	out = { key:[] for key in header }
	for row in data:
		for key, val in zip(header, row):
			out[key].append(val)
	out = { key:np.array(values) for key,values in out.items() }
	return out

def desc(headers, data_dict):
	klen = max( len(k) for k in headers )
	for key in headers:
		vals = data_dict[key]
		print(key.ljust(klen), '\tn=%d'%vals.shape, '\tdtype=%r'%vals.dtype)

def make_data(data_dict, **fields):
	output = {}
	for k,v in fields.items():
		if isinstance(v, list):
			val = np.hstack([ data_dict[_v][:,None] for _v in v ])
		else:
			val = data_dict[v]
		output[k] = val
		output['%s_label' % k] = v
	return output

if __name__ == '__main__':

	header, data = parse_data('compiled_data.csv')
	data_dict = data_to_dict(header, data)
	desc(header, data_dict)

	output_d   = {}
	output_h = []

	race_map = {
		'Amarela'  : 0,
		'Branca'   : 1,
		'Negra'    : 2,
		'Parda'    : 3,
		'Vermelha' : 4
	}
	output_d['race'] = np.array([ race_map[v] for v in data_dict['Race'] ])
	output_h.append('race')

	sex_map = {
		'F' : 0,
		'M' : 1
	}
	output_d['sex'] = np.array([ sex_map[v] for v in data_dict['Gender'] ])
	output_h.append('sex')

	header_map = {
		'SAT_Physics' 				: 'physics_exam',
		'SAT_Biology' 				: 'biology_exam',
		'SAT_History' 				: 'history_exam',
		'SAT_Second_Language' 		: 'second_language_exam',
		'SAT_Geography' 			: 'geography_exam',
		'SAT_Literature' 			: 'literature_exam',
		'SAT_Portuguese_and_Essay' 	: 'portuguese_exam',
		'SAT_Math' 					: 'math_exam',
		'SAT_Chemistry' 			: 'chemistry_exam'
	}
	for k,v in header_map.items():
		output_d[v] = data_dict[k]
		output_h.append(v)

	output_d['gpa'] = (data_dict['S01_Avg_Grade'] + data_dict['S02_Avg_Grade'] + data_dict['S03_Avg_Grade']) / 3
	output_h.append('gpa')

	D = make_data(output_d, X=output_h[2:-1], Y=output_h[-1], R=output_h[0], S=output_h[1])
	np.savez('brazil_with_race.npz', **D)
	

	


