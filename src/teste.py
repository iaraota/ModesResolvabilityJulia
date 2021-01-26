import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import pandas as pd
import scipy.integrate as integrate
import julia

path = "/home/iara/Doutorado/Julia/ModesResolvability/"

DataNResolv = {}
with h5py.File(path+"data/rayleigh/NumberModeHorizon_01.5.SXSBBH0593_crit.h5", "r") as f:
    # detector key

    for detector in list(f.keys()):
    	DataNResolv[detector] = {}
    	for num_par in list(f[detector].keys()):
    		DataNResolv[detector][num_par] = {}
    		for parameter in list(f[detector][num_par].keys()):
    			DataNResolv[detector][num_par][parameter] = np.array(f[detector][num_par][parameter])


def ContourPoints(X,Y,Z):
	# Compute contour points of data where Z assume binary values

	dfdx = lambda f, x, i, j: f[i+1][j] - f[i][j]#/(x[i+1, j] - x[i-1, j])
	dfdy = lambda f, x, i, j: f[i][j+1] - f[i][j]#/(y[i, j+1] - y[i, j-1])

	M = len(Z[:,1])
	N = len(Z[1,:])
	x = []
	y = []
	for i in range(1,M-1):
		for j in range(0,N-1):
			if abs(dfdx(Z, X, i, j))+abs(dfdy(Z, X, i, j)): # && Z[i,j] != 0
				x.append(X[i][j])
				y.append(Y[i][j])

	# Sort points according to the path
	
	idx_sort = np.argsort(x)	# sort x reversed
	x = np.array(x)[idx_sort][::-1]
	y = np.array(y)[idx_sort][::-1]

	## select (x,y) while y is increasing  (x is reversed, decreasing)
	i_right = 0
	i_aux = 0
	while True:
		if y[i_aux] <= y[i_aux + 1]:
			i_right = i_aux
			i_aux += 1
		else:
			break
	
	indices_y = [i_right]
	i_max = 0
	for i in range(i_right,len(y)):
		if y[i] >= y[indices_y[-1]] and x[i] < x[indices_y[-1]]:
			indices_y.append(i)

	
	x_right = x[0:i_right-1]
	y_right = y[0:i_right-1]
	for i in indices_y:
		np.append(x_right, x[i])
		np.append(y_right, y[i])

	np.delete(x, indices_y)
	np.delete(y, indices_y)
	x_left = x[i_right+1:-1]
	y_left = y[i_right+1:-1]

	idx_sort_yleft = np.argsort(y_left)
	x_left = np.array(x_left)[idx_sort_yleft]
	y_left = np.array(y_left)[idx_sort_yleft]
	
	x = np.concatenate((x_right, x_left))
	y = np.concatenate((y_right, y_left))

	# make curve smooth by deleting "zigzag" and repeted points
	
	indices = [1]
	while indices:
		indices = []
		for i in range(1,len(y)-1):
			if y[i] <= y[i-1] and y[i] <= y[i+1]:
				np.append(indices, i)

		np.delete(x, indices)
		np.delete(y, indices)

	return x, y

detector = 'LIGO'
num_pars = '6'
k = 4
X, Y = DataNResolv[detector][num_pars]["mass"], DataNResolv[detector][num_pars]["redshift"]
x, y = ContourPoints(X, Y, DataNResolv[detector][num_pars][str(k)+"modes"])	

print(all(x[i] >= x[i+1] for i in range(len(x)-1)))