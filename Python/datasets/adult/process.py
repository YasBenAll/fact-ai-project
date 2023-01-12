import numpy as np

with open('datasets/adult/adult_norm.csv', 'r') as f:
	raw = list(f)
labels, *raw = [ d.strip().split(',') for d in raw if d.find('?') == -1 ]
data = { k:v for k,v in zip(labels, np.array(raw).T)}

numerical_fields = []
categorical_fields = []
for k,v in data.items():
	try:
		data[k] = data[k].astype(float)
		min, max = data[k].min(), data[k].max()
		# data[k] = (data[k]-min) / (max-min)
		data[k] = (data[k]/max).round(9)
		numerical_fields.append(k)
	except:
		categorical_fields.append(k)

for k in categorical_fields:
	data[k] = { ('is_%s'%v) : 1*(data[k]==v) for v in np.unique(data[k]) }

data_flat = {}
labels_flat = []
for k,v in data.items():
	if isinstance(v, dict):
		for _k,_v in v.items():
			new_k = '%s:%s' % (k,_k)
			data_flat[new_k] = _v
			labels_flat.append(new_k)
	else:
		data_flat[k] = v
		labels_flat.append(k)
nontriv_dat_uvals = {k:np.unique(v) for k,v in data_flat.items() if len(np.unique(v)) > 2 }

with open('adult_norm.csv', 'r') as f:
	raw = list(f)
raw = [ d.strip().split(',') for d in raw ]
ref = np.array(raw).astype(float).T
nontriv_ref_inds  = [i for i in range(ref.shape[0]) if len(np.unique(ref[i])) > 2]
nontriv_ref_uvals = [np.unique(ref[i]) for i in range(ref.shape[0]) if len(np.unique(ref[i])) > 2]


ref_uvals = { len(v):(i,v) for i,v in zip(nontriv_ref_inds,nontriv_ref_uvals) }
dat_uvals = { len(v):(k,v) for k,v in nontriv_dat_uvals.items() }
matches = { dat_uvals[i][0]:{'ref_ind':ref_uvals[i][0], 'ref':ref_uvals[i][1], 'dat':dat_uvals[i][1].round(9)} for i in ref_uvals.keys() }
matches = { k:v for k,v in matches.items() if all(v['ref']==v['dat'])}
matches = { dat_uvals[i][0]:{'ref_ind':ref_uvals[i][0], 'ref':ref_uvals[i][1], 'dat':dat_uvals[i][1].round(9)} for i in ref_uvals.keys() }
matches['income:is_>50K'] = {'ref_ind':41}
matches['gender:is_Male'] = {'ref_ind':0}

ref_tuples = np.array([ ref[v['ref_ind']] for k,v in matches.items() ]).T
dat_tuples = np.array([ data_flat[k] for k,v in matches.items() ]).T
ref_dat_map = []
for i in range(ref_tuples.shape[0]):
	dat_inds = np.where((ref_tuples[i]==dat_tuples).all(1))[0]
	if len(dat_inds) == 1:
		ref_dat_map.append((i,dat_inds[0]))
		print(len(ref_dat_map))
	elif len(dat_inds) == 0:
		print('Ref indec %d not found in data.' % i)
ref_dat_map = np.array(ref_dat_map)

filtered_ref = ref[:,ref_dat_map[:,0]].T
filtered_dat = { k:v[ref_dat_map[:,1]] for k,v in data_flat.items() }

cands = {k:np.where((filtered_dat[k]==filtered_ref.T).all(1)) for k in filtered_dat.keys()}
cands = {k:v[0] for k,v in cands.items() if len(v[0])>0}
assert all(len(v)==1 for v in cands.values()), 'error'
inds = {v[0]:k for k,v in cands.items()}


labels = [inds[i] for i in range(len(inds))]




labels = 'gender:is_Male,workclass:is_Without-pay,workclass:is_State-gov,workclass:is_Self-emp-not-inc,workclass:is_Self-emp-inc,workclass:is_Private,workclass:is_Local-gov,marital-status:is_Widowed,marital-status:is_Separated,marital-status:is_Never-married,marital-status:is_Married-spouse-absent,marital-status:is_Married-civ-spouse,marital-status:is_Married-AF-spouse,occupation:is_Transport-moving,occupation:is_Tech-support,occupation:is_Sales,occupation:is_Protective-serv,occupation:is_Prof-specialty,occupation:is_Priv-house-serv,occupation:is_Other-service,occupation:is_Machine-op-inspct,occupation:is_Handlers-cleaners,occupation:is_Farming-fishing,occupation:is_Exec-managerial,occupation:is_Craft-repair,occupation:is_Armed-Forces,relationship:is_Wife,relationship:is_Unmarried,relationship:is_Own-child,relationship:is_Other-relative,relationship:is_Not-in-family,race:is_White,race:is_Other,race:is_Black,race:is_Asian-Pac-Islander,native-country:is_United-States,age,educational-num,capital-gain,capital-loss,hours-per-week,income:is_>50K'