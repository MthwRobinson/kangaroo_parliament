from __future__ import division
import numpy as np 
import pandas as pd
from pandas import DataFrame
import statsmodels.formula.api as sm
import pickle

def load_data():
	data = pd.read_csv('/home/matt/example_code/mortality.csv')
	data['SO2'] = data['SO@']
	data = data[['PREC', 'EDUC', 'NONW', 'SO2', 'MORT']]
	return data

def fit_model(data):
	formula =  'MORT ~ PREC + np.power(PREC,2) + EDUC + '
	formula += 'np.power(EDUC,2) + NONW + np.power(PREC,2)'
	fm1c = sm.ols(formula = formula, data = data)
	res_fm1c = fm1c.fit()
	return res_fm1c

if __name__ == '__main__':
	print 'loading data ... '
	data = load_data()
	print 'fitting model ... '
	model = fit_model(data)
	print 'writing model to disk ...'
	pickle.dump(model, 
		open('/home/matt/kangaroo_parliament/roo/roo/models/model.p', 'wb'))
	print 'done!'


