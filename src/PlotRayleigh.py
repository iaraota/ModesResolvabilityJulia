import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import scipy.integrate as integrate

def luminosity_distance(redshift):
    ## Luminosity distance as as function of the redshift
    # cosmological constans
    # values from https://arxiv.org/pdf/1807.06209.pdf
    h = 0.6796
    H_0 = h*100*1e+3 # Huble constant m s^-1 Mpc^-1
    clight = 2.99792458e8 # speed of lightm s^-1
    D_H = clight/H_0 # Huble distance

    Ω_M = 0.315
    Ω_Λ = 1-Ω_M
    Ω_K = 0.0

    E = lambda z: np.sqrt(Ω_M*(1+z)**3 + Ω_K*(1+z)**2 + Ω_Λ)

    D_C = D_H*integrate.quad(lambda x: 1/E(x), 0, redshift)[0]
    D_L = (1 + redshift)*D_C

    return D_L

def SelectSimulation(q_mass):
	# TODO: add all simulations/ optimize simulation selection
	if q_mass == 1.5:
		label = "01.5.SXSBBH0593"
	elif q_mass == 5.0:
		label = "05.0.SXSBBH0107"
	elif q_mass == 10.0:
		label = "10.0.SXSBBH1107"
	return label

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
		if y[i_aux] <= y[i_aux + 1] and i_aux < len(y) -2:
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
	x_left = np.array(x_left)[idx_sort_yleft][::-1]
	y_left = np.array(y_left)[idx_sort_yleft][::-1]
	
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


def PlotResolvableModesContourAll(q_mass):
	plt.rcParams["mathtext.fontset"] = "cm"
	plt.rcParams["font.family"] = "STIXGeneral"
	plt.rcParams["figure.figsize"] = [20, 8]  # plot image size

	SMALL_SIZE = 20
	MEDIUM_SIZE = 28
	BIGGER_SIZE = 32
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)

	colors = {2: "black", 3: "tab:red", 4: "tab:green", 5: "tab:blue"}

	label = SelectSimulation(q_mass)

	DataNResolv = {}
	path = "/home/iara/Doutorado/Julia/ModesResolvability/"
	with h5py.File(path+"data/rayleigh/NumberModeHorizon_"+label+".h5", "r") as f:
	    # detector key

	    for detector in list(f.keys()):
	    	DataNResolv[detector] = {}
	    	for num_par in list(f[detector].keys()):
	    		DataNResolv[detector][num_par] = {}
	    		for parameter in list(f[detector][num_par].keys()):
	    			DataNResolv[detector][num_par][parameter] = np.array(f[detector][num_par][parameter])

	# import GW catalog
	catalog_file = path+"data/catalog_mass_dl_z.csv"
	catalog = pd.read_csv(catalog_file)


	for num_pars in ["4", "6"]:
		plt.close("all")
		fig, ax1 = plt.subplots()

		# plot catalog
		ax1.scatter(catalog["final_mass"], catalog["redshift"], marker = "*", s = 150, 
					color = "gold", edgecolors = "darkgoldenrod")
		
		ax1.set_xscale("log")  
		ax1.set_yscale("log")  
		ax1.set_ylim(1e-2, 1e3)
		ax2 = ax1.twinx()
		mn, mx = ax1.get_ylim()
		ax2.set_yscale("log")  
		ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
		ax2.set_ylabel("distância luminosa [Gpc]")
	
		ax1.set_ylabel(r"redshift, $z$")
		ax1.set_xlabel(r"massa final $[M_{\odot}]$")
		legends_lines = []
		legends_labels = []
		for detector in ["LIGO", "CE", "ET", "LISA"]:
			if detector == "LIGO":
				ls = "solid"
			elif detector == "ET":
				ls = "dashed"
			elif detector == "CE":
				ls = "dotted"
			elif detector == "LISA":
				ls = "solid"

			X, Y = DataNResolv[detector][num_pars]["mass"], DataNResolv[detector][num_pars]["redshift"]
			#X = (1 .+ Y).*X

			for k in [2,3,4,5]:
				if np.any(DataNResolv[detector][num_pars][str(k)+"modes"]):
					x, y = ContourPoints(X, Y, DataNResolv[detector][num_pars][str(k)+"modes"])					
					if detector == "LISA":
						ax1.plot(x,y, label = str(k)+" modes", lw = 3, ls = ls, color = colors[k])
					else:
						ax1.plot(x,y, lw = 3, ls = ls, color = colors[k])
					#ax1.fill_between(x,y, 0, alpha = 0.5, color = colors[k])
				#ax1.contourf(X,Y,  DataNResolv[detector][num_pars][string(k)*"modes"], levels=[.9,2], alpha = 0.5)		


		ax1.set_xlim(1e1, 1e9)        
		ax1.legend()
		
		extra = mpl.lines.Line2D([0], [0], color="white")
		# legend_extra = plt.legend([extra], [r"$q = {}$".format(q_mass)], handlelength = 0, fontsize = SMALL_SIZE, frameon = false, 	bbox_to_anchor=(0.08, 0.999))
		# plt.gca().add_artist(legend_extra)

		# ax1.set_title("Spectroscopy horizon, "*num_pars*" parameters")

		fig.tight_layout()
		# plt.savefig("figs/rayleigh/all_Nmodes_"*num_pars*"_"*label*"_crit.pdf")
		plt.show()

PlotResolvableModesContourAll(1.5)
