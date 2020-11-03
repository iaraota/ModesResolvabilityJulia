using PyPlot, HDF5, DelimitedFiles, DataFrames, LaTeXStrings, QuadGK, StatsBase
using PyCall,Dierckx
@pyimport scipy.interpolate as si
mpl = pyimport("matplotlib")
include("AllFunctions.jl")
using .Quantities

function ImportDataGrid(file_name, k_mode, k_num, min, max)	
    files = Dict()

    for i in min:max
		files["e"*string(i)] = h5open(file_name*"e"*string(i)*".h5") do file
	        read(file, k_mode) 
	    end
	end

	data = reduce(merge, collect(values(files)))
	
    M_f = zeros(0)
    
	for mass in keys(data)
		append!(M_f, parse(Float64,mass))
	end
	M_f = sort(M_f)

	redshift = data[string(M_f[1])][k_num][:,1]

	M, N = length(M_f), length(redshift)
	
	df, σf, dtau, σtau = zeros(M, N), zeros(M, N), zeros(M, N), zeros(M, N)
    dfQ, σfQ, dQ, σQ = zeros(M, N), zeros(M, N), zeros(M, N), zeros(M, N)

	for i in 1:M
		for j in 1:N
			df[i, j] = data[string(M_f[i])][k_num][j, 2]
			σf[i, j] = data[string(M_f[i])][k_num][j, 3]
			dtau[i, j] = data[string(M_f[i])][k_num][j, 4]
			σtau[i, j] = data[string(M_f[i])][k_num][j, 5]
			dfQ[i, j] = data[string(M_f[i])][k_num][j, 6]
			σfQ[i, j] = data[string(M_f[i])][k_num][j, 7]
			dQ[i, j] = data[string(M_f[i])][k_num][j, 8]
			σQ[i, j] = data[string(M_f[i])][k_num][j, 9]
		end
	end

	X, Y = meshgrid(M_f, redshift)
	Z_ftau, Z_tau, Z_fQ, Z_Q = df./σf, dtau./σtau, dfQ./σfQ, dQ./σQ
	Z_tau_crit, Z_tau_both, Z_Q_crit, Z_Q_both = zeros(M, N), zeros(M, N), zeros(M, N), zeros(M, N)

	for i in 1:M
		for j in 1:N
			if Z_ftau[i, j] > 1 || Z_tau[i, j] > 1
				Z_tau_crit[i, j] = 1
			else
				Z_tau_crit[i,j] = 0
			end
			if Z_ftau[i,j] > 1 && Z_tau[i,j] > 1
				Z_tau_both[i,j] = 1
			else
				Z_tau_both[i,j] = 0
			end
			if Z_fQ[i,j] > 1 || Z_Q[i,j] > 1
				Z_Q_crit[i,j] = 1
			else
				Z_Q_crit[i,j] = 0
			end
			if Z_fQ[i,j] > 1 && Z_Q[i,j] > 1
				Z_Q_both[i,j] = 1
			else
				Z_Q_both[i,j] = 0
			end
		end
	end
	return X, Y, Z_ftau, Z_tau, Z_fQ, Z_Q, Z_tau_crit, Z_tau_both, Z_Q_crit, Z_Q_both

end

function ImportModesGrid(file_name, min, max, choose_modes = "All", merge_221 = false)
	if choose_modes == "All"	
		modes = ["(2,2,0)+(2,2,1) I", "(2,2,0)+(3,3,0)", "(2,2,0)+(4,4,0)", "(2,2,0)+(2,1,0)",
		"(2,2,1) I+(3,3,0)", "(2,2,1) I+(4,4,0)", "(2,2,1) I+(2,1,0)",
		"(3,3,0)+(2,1,0)", "(3,3,0)+(4,4,0)", "(4,4,0)+(2,1,0)", "(2,2,0)+(2,2,1) II", 
		"(2,2,1) II+(3,3,0)", "(2,2,1) II+(4,4,0)", "(2,2,1) II+(2,1,0)"]
	elseif choose_modes == "(2,2,0)"
		modes = ["(2,2,0)+(2,2,1) I", "(2,2,0)+(2,2,1) II", "(2,2,0)+(3,3,0)", "(2,2,0)+(4,4,0)", "(2,2,0)+(2,1,0)",]
	elseif choose_modes == "(2,2,1)"
		modes = ["(2,2,1) I+(4,4,0)", "(2,2,1) I+(3,3,0)", "(2,2,0)+(2,2,1) I", "(2,2,1) I+(2,1,0)",
		"(2,2,1) II+(4,4,0)", "(2,2,1) II+(3,3,0)", "(2,2,0)+(2,2,1) II", "(2,2,1) II+(2,1,0)"]
	elseif choose_modes == "(3,3,0)"
		modes = ["(2,2,0)+(3,3,0)", "(2,2,1) I+(3,3,0)", "(2,2,1) II+(3,3,0)", "(3,3,0)+(2,1,0)", "(3,3,0)+(4,4,0)"]
	elseif choose_modes == "(4,4,0)"
		modes = ["(2,2,0)+(4,4,0)", "(2,2,1) I+(4,4,0)", "(2,2,1) II+(4,4,0)", "(4,4,0)+(2,1,0)", "(3,3,0)+(4,4,0)"]
	elseif choose_modes == "(2,1,0)"
		modes = ["(2,2,0)+(2,1,0)", "(2,2,1) I+(2,1,0)", "(2,2,1) II+(2,1,0)", "(3,3,0)+(2,1,0)", "(4,4,0)+(2,1,0)"]
	end	

	num_par = ["4","6"]
	ModesData = Dict()
	for k_mode in modes
		ModesData[k_mode] = Dict()
		for k_num in num_par
			ModesData[k_mode][k_num]  = ImportDataGrid(file_name, k_mode, k_num, min, max)	
		end				
	end
	if merge_221 == true
		if choose_modes == "(2,2,0)"
			ModesData["(2,2,0)+(2,2,1)"] = Dict()
		elseif choose_modes == "(2,2,1)"
			ModesData["(2,2,0)+(2,2,1)"] = Dict()
			for harmonics in ["(3,3,0)", "(4,4,0)", "(2,1,0)"]
				ModesData["(2,2,1)+"*harmonics] = Dict()
			end
		else
			ModesData["(2,2,1)+"*choose_modes] = Dict()
		end
		
		for num_pars in num_par
			if choose_modes == "(2,2,0)"
				ModesData["(2,2,0)+(2,2,1)"][num_pars] = (ModesData["(2,2,0)+(2,2,1) I"][num_pars] .+ ModesData["(2,2,0)+(2,2,1) II"][num_pars])./2
			elseif choose_modes == "(2,2,1)"
				ModesData["(2,2,0)+(2,2,1)"][num_pars] = (ModesData["(2,2,0)+(2,2,1) I"][num_pars] .+ ModesData["(2,2,0)+(2,2,1) II"][num_pars])./2
				for harmonics in ["(3,3,0)", "(4,4,0)", "(2,1,0)"]
					ModesData["(2,2,1)+"*harmonics][num_pars] = (ModesData["(2,2,1) I+"*harmonics][num_pars] .+ ModesData["(2,2,1) II+"*harmonics][num_pars])./2
				end
			else
				ModesData["(2,2,1)+"*choose_modes][num_pars] = (ModesData["(2,2,1) I+"*choose_modes][num_pars] .+ ModesData["(2,2,1) II+"*choose_modes][num_pars])./2
			end
		end
		if choose_modes == "(2,2,0)"
			delete!(ModesData, "(2,2,0)+(2,2,1) I")
			delete!(ModesData, "(2,2,0)+(2,2,1) II")
		elseif choose_modes == "(2,2,1)"
			delete!(ModesData, "(2,2,0)+(2,2,1) I")
			delete!(ModesData, "(2,2,0)+(2,2,1) II")
			for harmonics in ["(3,3,0)", "(4,4,0)", "(2,1,0)"]
				delete!(ModesData, "(2,2,1) I+"*harmonics)
				delete!(ModesData, "(2,2,1) II+"*harmonics)
			end
		else
			delete!(ModesData, "(2,2,1) I+"*choose_modes)
			delete!(ModesData, "(2,2,1) II+"*choose_modes)
		end

	end
	return ModesData	
end

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function ContourPoints(X,Y,Z)
	# interpolate line for binary contour
	dfdx(f, x, i, j) = f[i+1, j] - f[i,j]#/(x[i+1, j] - x[i-1, j])
	dfdy(f, y, i, j) = f[i, j+1] - f[i,j]#/(y[i, j+1] - y[i, j-1])

	M = length(Z[:,1])
	N = length(Z[1,:])
	x = zeros(0)
	y = zeros(0)
	for i in 1:M-1
		for j in 1:N-1
			if abs(dfdx(Z, X, i, j))+abs(dfdy(Z, X, i, j)) != 0.0 # && Z[i,j] != 0
				append!(x, X[i,j])
				append!(y, Y[i,j])
			end
		end
	end
	indices = [1]
	# makie curve smooth by deleting "zigzag"
	#=
	while indices != []
		indices = Int[]
		for i in 2:length(y)-1
			if y[i] < y[i-1] && y[i] < y[i+1]
				append!(indices, i)
			end
		end
		deleteat!(x, indices)
		deleteat!(y, indices)
	end
	=#
	return x, y	
end

function NumberResolvableModes(DataModes,num_par="4")
	higher_modes = ["(2,2,1)", "(3,3,0)", "(4,4,0)", "(2,1,0)"]

	ResolveData = Dict()
	X, Y = 0,0
	for (key, value) in DataModes
		# TODO: choose critical or both, tau or Q
		X = value[num_par][1]
		Y = value[num_par][2]
		ResolveData[key] = value[num_par][8]
	end
	
	# consider the mean value between (2,2,1) two methods
	ResolveData["(2,2,0)+(2,2,1)"] = (ResolveData["(2,2,0)+(2,2,1) I"] .+ ResolveData["(2,2,0)+(2,2,1) II"])./2
	delete!(ResolveData, "(2,2,0)+(2,2,1) I")
	delete!(ResolveData, "(2,2,0)+(2,2,1) II")
	for k in ["(3,3,0)", "(4,4,0)", "(2,1,0)"]
		ResolveData["(2,2,1)+"*k] = (ResolveData["(2,2,1) I+"*k] .+ ResolveData["(2,2,1) II+"*k])./2
		delete!(ResolveData, "(2,2,1) I+"*k)
		delete!(ResolveData, "(2,2,1) II+"*k)
	end

	Resolve_220 = Dict()
	for modes in higher_modes
		Resolve_220["(2,2,0)+"*modes] = ResolveData["(2,2,0)+"*modes] 
	end
	M = length(ResolveData["(2,2,0)+(2,2,1)"][:,1])
	N = length(ResolveData["(2,2,0)+(2,2,1)"][1,:])
	# resolve two modes
	Resolve2modes = zeros(M,N)
	for i in 1:M
		for j in 1:N
			if any(x -> x == 1, [y[i,j] for y in values(Resolve_220)])
				Resolve2modes[i,j] = 1
			end
		end
	end

	# resolve 3 modes
	Resolve3modes = zeros(M,N)
	Sum3Modes = Dict()
	aux = String[]
	for mode_1 in higher_modes
		push!(aux, mode_1)
		for mode_2 in  filter(x-> x ∉ aux, higher_modes)
			Sum3Modes[mode_1*"+"*mode_2] = ResolveData["(2,2,0)+"*mode_1] .+ ResolveData["(2,2,0)+"*mode_2] .+ ResolveData[mode_1*"+"*mode_2]
		end 
	end
	for i in 1:M
		for j in 1:N
			if any(x -> x == binomial(3,2), [y[i,j] for y in values(Sum3Modes)])
				Resolve3modes[i,j] = 1
			end
		end
	end

	# resolve 4 modes
	Resolve4modes = zeros(M,N)
	Sum4Modes = Dict()
	while higher_modes != []
		for mode_1 in higher_modes
			aux = String[]
			push!(aux, mode_1)
			for mode_2 in  filter(x-> x ∉ aux, higher_modes)
				push!(aux, mode_2)
				for mode_3 in filter(x -> x ∉ aux, higher_modes)
					modes = sort([mode_1, mode_2, mode_3]) 
					if modes[1] == "(2,1,0)"
						modes[1] = modes[2]
						modes[2] = modes[3]
						modes[3] = "(2,1,0)"
					end
					Sum4Modes[modes[1]*"+"*modes[2]*"+"*modes[3]] = (
						ResolveData["(2,2,0)+"*modes[1]] .+ ResolveData["(2,2,0)+"*modes[2]] 
						.+ ResolveData["(2,2,0)+"*modes[3]] .+ ResolveData[modes[1]*"+"*modes[2]] 
						.+ ResolveData[modes[1]*"+"*modes[3]] .+ ResolveData[modes[2]*"+"*modes[3]])
				end
			end 
		filter!(x -> x != mode_1, higher_modes)
		end
	end

	for i in 1:M
		for j in 1:N
			if any(x -> x == binomial(4,2), [y[i,j] for y in values(Sum4Modes)])
				Resolve4modes[i,j] = 1
			end
		end
	end

	# resolve 5 modes
	Sum5Modes = zeros(M,N)
	Resolve5modes = zeros(M,N)
	for values in values(ResolveData)
		Sum5Modes .+= values
	end
	for i in 1:M
		for j in 1:N
			if any(x -> x == binomial(5,2), Sum5Modes[i,j])
				Resolve5modes[i,j] = 1
			end
		end
	end
	
	return X, Y, Resolve2modes, Resolve3modes, Resolve4modes, Resolve5modes
end

function NewDataNumberResolvable(detector)
	if detector == "LIGO"
		min = 1
		max = 4
		smooth = 1
	elseif detector == "ET" || detector == "CE"
		min = 1
		max = 4
	elseif detector == "LISA"
		min = 4
		max = 9
	else
		error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
	end
	file_path = "/home/iara/Desktop/rayleigh/data_all/grande/FisherErrorsRedshift_01.5.SXSBBH0593_"*detector*"_FH_"
	DataModes = ImportModesGrid(file_path, min, max)
	if ! isdir("data/rayleigh/")
		mkdir("data/rayleigh/")
	end

	for num_pars in ["4", "6"]
		NumResolv = Dict()
		X, Y, NumResolv[2], NumResolv[3], NumResolv[4], NumResolv[5] = NumberResolvableModes(DataModes, num_pars)
		h5open("data/rayleigh/NumberModeHorizon.h5", "cw") do file
			write(file, detector*"/"*num_pars*"/mass", X)
			write(file, detector*"/"*num_pars*"/redshift", Y)
			write(file, detector*"/"*num_pars*"/2modes", NumResolv[2])
			write(file, detector*"/"*num_pars*"/3modes", NumResolv[3])
			write(file, detector*"/"*num_pars*"/4modes", NumResolv[4])
			write(file, detector*"/"*num_pars*"/5modes", NumResolv[5])
		end


		XX, YY = 0,0
		ResolveData = Dict()
		for (key, value) in DataModes
			# TODO: choose critical or both, tau or Q
			X = value[num_pars][1]
			Y = value[num_pars][2]
			ResolveData[key] = value[num_pars][8]
		end
		
		# consider the mean value between (2,2,1) two methods
		ResolveData["(2,2,0)+(2,2,1)"] = (ResolveData["(2,2,0)+(2,2,1) I"] .+ ResolveData["(2,2,0)+(2,2,1) II"])./2
		delete!(ResolveData, "(2,2,0)+(2,2,1) I")
		delete!(ResolveData, "(2,2,0)+(2,2,1) II")
		for k in ["(3,3,0)", "(4,4,0)", "(2,1,0)"]
			ResolveData["(2,2,1)+"*k] = (ResolveData["(2,2,1) I+"*k] .+ ResolveData["(2,2,1) II+"*k])./2
			delete!(ResolveData, "(2,2,1) I+"*k)
			delete!(ResolveData, "(2,2,1) II+"*k)
		end
		h5open("data/rayleigh/ModeHorizon.h5", "cw") do file
			write(file, detector*"/"*num_pars*"/mass", XX)
			write(file, detector*"/"*num_pars*"/redshift", YY)
		end
		for (k,v) in ResolveData
			h5open("data/rayleigh/ModeHorizon.h5", "cw") do file
				write(file, detector*"/"*num_pars*"/"*k, v)
			end
		end
	end

end


function PlotResolvableModes(detector)
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)

	if detector == "LIGO"
		min = 1
		max = 4
		smooth = 1
	elseif detector == "ET" || detector == "CE"
		min = 1
		max = 4
	elseif detector == "LISA"
		min = 4
		max = 9
	else
		error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
	end
	file_path = "/home/iara/Desktop/rayleigh/data_all/FisherErrorsRedshift_01.5.SXSBBH0593_"*detector*"_FH_"
	DataModes = ImportModesGrid(file_path, min, max)

	for num_pars in ["4", "6"]
		NumResolv = Dict()
		X, Y, NumResolv[2], NumResolv[3], NumResolv[4], NumResolv[5] = NumberResolvableModes(DataModes, num_pars)

        close("all")
	    fig, ax1 = subplots()
        
		ax1.set_xscale("log")  
        ax1.set_yscale("log")  
        ax1.set_ylim(1e-2, 1e2)
        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
        ax2.set_ylabel("Luminosity distance [Gpc]")
    
        ax1.set_ylabel("redshift")
        ax1.set_xlabel(L"final mass $[M_\odot]$")
        for k in [2,3,4,5]
			x, y = ContourPoints(X, Y, NumResolv[k])
			
            ax1.plot(x,y, label = string(k)*" modes", lw = 3, ls = "-")
			ax1.fill_between(x,y, 0, alpha = 0.5)
        end
        ax1.set_xlim(left = 1e1)
        
        ax1.legend()

        ax1.set_title(detector*", "*num_pars*" parameters")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/"*detector*"_Nmodes_"*num_pars*"_menor.pdf")
	end
end

function PlotResolvableModesAll()
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)

	for num_pars in ["4", "6"]
		close("all")
		fig, ax1 = subplots()
		
		ax1.set_xscale("log")  
		ax1.set_yscale("log")  
		ax1.set_ylim(1e-2, 1e2)
		ax2 = ax1.twinx()
		mn, mx = ax1.get_ylim()
		ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
		ax2.set_ylabel("Luminosity distance [Gpc]")
	
		ax1.set_ylabel("redshift")
		ax1.set_xlabel(L"final mass $[M_\odot]$")
		for detector in ["LIGO", "CE", "ET", "LISA"]
			if detector == "LIGO"
				min = 1
				max = 4
				smooth = 1
			elseif detector == "ET" || detector == "CE"
				min = 1
				max = 4
			elseif detector == "LISA"
				min = 5
				max = 9
			else
				error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
			end
			
			file_path = "/home/iara/Desktop/rayleigh/data_all/FisherErrorsRedshift_01.5.SXSBBH0593_"*detector*"_FH_"
			DataModes = ImportModesGrid(file_path, min, max)
			
			ls = Dict("ET" => ":", "CE" => "--", "LISA" => "-.")

			NumResolv = Dict()
			X, Y, NumResolv[2], NumResolv[3], NumResolv[4], NumResolv[5] = NumberResolvableModes(DataModes, num_pars)

			for k in [2,3,4,5]
				x, y = ContourPoints(X, Y, NumResolv[k])
				if detector == "LIGO"
					ax1.plot(x,y, label = string(k)*" modes", lw = 3, ls = "-")
				else
					ax1.plot(x,y, lw = 3, ls = ls[detector])
				end				
				#ax1.fill_between(x,y, 0, alpha = 0.5)
			end
		end

		ax1.set_xlim(1e1, 1e9)
        
		ax1.legend()
		legend_extra = legend([extra], [L"$q = 1.5$"], handlelength = 0, fontsize = font_size -10, frameon = false, 	bbox_to_anchor=(0.09, 0.999))
		gca().add_artist(legend_extra)

        ax1.set_title(num_pars*" parameters")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/All_Nmodes_"*num_pars*".pdf")
	end
end


function PlotResolvableModesAllGrid()
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)
	c_221_l, c_33_l, c_44_l, c_21_l = "tab:red", "tab:blue", "tab:green", "tab:orange"
	

	cmap_f = Dict(
		2 => mpl.colors.ListedColormap([c_221_l]),
		3 => mpl.colors.ListedColormap([c_44_l]),
		4 => mpl.colors.ListedColormap([c_33_l]),
		5 => mpl.colors.ListedColormap([c_21_l])
		)
	cols = Dict(
		2 => c_221_l,
		3 => c_44_l,
		4 => c_33_l,
		5 => c_21_l
		)

	for num_pars in ["4", "6"]
		close("all")
		fig, ax1 = subplots()
		
		ax1.set_xscale("log")  
		ax1.set_yscale("log")  
		ax1.set_ylim(1e-2, 2e1)
		ax2 = ax1.twinx()
		mn, mx = ax1.get_ylim()
		ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
		ax2.set_ylabel("Luminosity distance [Gpc]")
	
		ax1.set_ylabel("redshift")
		ax1.set_xlabel(L"final mass $[M_\odot]$")
		legends_lines = []
		legends_labels = String[]
		for detector in ["LIGO", "CE", "ET", "LISA"]
			if detector == "LIGO"
				min = 1
				max = 4
				smooth = 1
				ls = "solid"
			elseif detector == "ET" || detector == "CE"
				min = 1
				max = 4
				ls = "dashed"
				if detector == "CE"
					ls = "dotted"
				end
			elseif detector == "LISA"
				min = 5
				max = 9
				ls = "solid"
			else
				error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
			end
			
			file_path = "/home/iara/Desktop/rayleigh/data_all/FisherErrorsRedshift_01.5.SXSBBH0593_"*detector*"_FH_"
			DataModes = ImportModesGrid(file_path, min, max)
			
			NumResolv = Dict()
			X, Y, NumResolv[2], NumResolv[3], NumResolv[4], NumResolv[5] = NumberResolvableModes(DataModes, num_pars)

			for k in [2,3,4,5]
				Z = NumResolv[k]

				#ax1.contourf(X,Y, Z, levels=[.9,2], cmap = cmap_f[k], alpha = 0.5)		
				ax1.contour(X,Y, Z, levels = 1, cmap = cmap_f[k], linewidths = 3, linestyles = ls)	
				
				if detector == "LIGO"
					push!(legends_lines, mpl.lines.Line2D([0], [0], color=cols[k], linewidth=3))
					push!(legends_labels, string(k)*" modes")
				end				
				#ax1.fill_between(x,y, 0, alpha = 0.5)
			end
		end

		ax1.set_xlim(1e1, 1e9)
        
        extra = mpl.lines.Line2D([0], [0], color="white")
		legend_extra = legend([extra], [L"$q = 1.5$"], handlelength = 0, fontsize = SMALL_SIZE, frameon = false, 	bbox_to_anchor=(0.09, 0.999))
		ax1.legend(legends_lines,legends_labels, loc = "upper right")
		
		gca().add_artist(legend_extra)

        ax1.set_title("Spectroscopy horizon, "*num_pars*" parameters")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/All_Nmodes_"*num_pars*".pdf")
	end
end

function PlotNumberResolvableModesAllGrid()
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [20, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)
	c_221_l, c_33_l, c_44_l, c_21_l = "tab:red", "tab:blue", "tab:green", "tab:orange"
	

	cmap_f = Dict(
		2 => mpl.colors.ListedColormap([c_221_l]),
		3 => mpl.colors.ListedColormap([c_44_l]),
		4 => mpl.colors.ListedColormap([c_33_l]),
		5 => mpl.colors.ListedColormap([c_21_l])
		)
	cols = Dict(
		2 => c_221_l,
		3 => c_44_l,
		4 => c_33_l,
		5 => c_21_l
		)
	DataNResolv = h5open("data/rayleigh/NumberModeHorizon.h5") do file
		read(file) 
	end
	for num_pars in ["4", "6"]
		close("all")
		fig, ax1 = subplots()
		
		ax1.set_xscale("log")  
		ax1.set_yscale("log")  
		ax1.set_ylim(1e-2, 1e2)
		ax2 = ax1.twinx()
		mn, mx = ax1.get_ylim()
		ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
		ax2.set_ylabel("Luminosity distance [Gpc]")
	
		ax1.set_ylabel("redshift")
		ax1.set_xlabel(L"final mass $[M_\odot]$")
		legends_lines = []
		legends_labels = String[]
		for detector in ["LIGO"]#, "CE", "ET", "LISA"]
			if detector == "LIGO"
				ls = "solid"
			elseif detector == "ET" 
				ls = "dashed"
			elseif detector == "CE"
				ls = "dotted"
			elseif detector == "LISA"
				ls = "solid"
			else
				error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
			end
			X, Y = DataNResolv[detector][num_pars]["mass"], DataNResolv[detector][num_pars]["redshift"]
			#X = (1 .+ Y).*X

			for k in [2,3,4,5]
				Z = DataNResolv[detector][num_pars][string(k)*"modes"]
				#ax1.contourf(X,Y, Z, levels=[.9,2], cmap = cmap_f[k], alpha = 0.5)		
				ax1.contour(X,Y, Z, levels = 1, cmap = cmap_f[k], linewidths = 3, linestyles = ls)	
				
				if detector == "LIGO"
					push!(legends_lines, mpl.lines.Line2D([0], [0], color=cols[k], linewidth=3))
					push!(legends_labels, string(k)*" modes")
				end				
				#ax1.fill_between(x,y, 0, alpha = 0.5)
			end
		end

		ax1.set_xlim(1e1, 1e9)
        
        extra = mpl.lines.Line2D([0], [0], color="white")
		legend_extra = legend([extra], [L"$q = 1.5$"], handlelength = 0, fontsize = SMALL_SIZE, frameon = false, 	bbox_to_anchor=(0.055, 1))
		ax1.legend(legends_lines,legends_labels, loc = "upper right")
		
		gca().add_artist(legend_extra)

        ax1.set_title("Spectroscopy horizon, "*num_pars*" parameters")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/All_Nmodes_"*num_pars*"_new.pdf")
	end
end


function PlotHorizon2modes(detector, mode)
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)

	if detector == "LIGO"
		min = 1
		max = 4
		smooth = 1
	elseif detector == "ET" || detector == "CE"
		min = 1
		max = 4
	elseif detector == "LISA"
		min = 4
		max = 9
	else
		error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
	end
	file_path = "/home/iara/Desktop/rayleigh/data_all/z100/FisherErrorsRedshift_01.5.SXSBBH0593_"*detector*"_FH_"
	DataModes = ImportModesGrid(file_path, min, max, mode, true)
	

	for num_pars in ["4", "6"]
		NumResolv = Dict()
        close("all")
	    fig, ax1 = subplots()
        
		ax1.set_xscale("log")  
        ax1.set_yscale("log")  
        ax1.set_ylim(1e-2, 1e2)
        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
        ax2.set_ylabel("Luminosity distance [Gpc]")
    
        ax1.set_ylabel("redshift")
        ax1.set_xlabel(L"final mass $[M_\odot]$")
		for (k,v) in DataModes
			#ax1.contour(v[num_pars][1], v[num_pars][2], v[num_pars][8], levels = 2)
			x, y = ContourPoints(v[num_pars][1], v[num_pars][2], v[num_pars][8])
			#x = x.*(1 .+ y)
            #ax1.plot(x,y, label = k, lw = 3, ls = "-")
            ax1.scatter(x,y, label = k, marker = ",", s= 1)
            #ax1.fill(x,y, alpha = 0.4)
			#ax1.fill_between(x,y, 0, alpha = 0.5)
        end
        ax1.set_xlim(left = 1e1)
        
        ax1.legend()

        ax1.set_title(detector*", "*num_pars*" parameters")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/"*detector*"_"*mode*"_"*num_pars*".pdf")
	end
end


function PlotHorizon2modesGrid(detector, mode)
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)
	c_221_l, c_33_l, c_44_l, c_21_l = "tab:red", "tab:blue", "tab:green", "tab:orange"
	

	cmap_f = Dict(
		"(2,2,0)+(2,2,1)" => mpl.colors.ListedColormap([c_221_l]),
		"(2,2,0)+(3,3,0)" => mpl.colors.ListedColormap([c_33_l]),
		"(2,2,1)+(3,3,0)" => mpl.colors.ListedColormap([c_33_l]),
		"(2,2,0)+(4,4,0)" => mpl.colors.ListedColormap([c_44_l]),
		"(2,2,1)+(4,4,0)" => mpl.colors.ListedColormap([c_44_l]),
		"(2,2,0)+(2,1,0)" => mpl.colors.ListedColormap([c_21_l]),
		"(2,2,1)+(2,1,0)" => mpl.colors.ListedColormap([c_21_l])
		)
	
	cols = Dict(
		"(2,2,0)+(2,2,1)" => c_221_l,
		"(2,2,0)+(3,3,0)" => c_33_l,
		"(2,2,1)+(3,3,0)" => c_33_l,
		"(2,2,0)+(4,4,0)" => c_44_l,
		"(2,2,1)+(4,4,0)" => c_44_l,
		"(2,2,0)+(2,1,0)" => c_21_l,
		"(2,2,1)+(2,1,0)" => c_21_l
		)

	if detector == "LIGO"
		min = 1
		max = 4
		smooth = 1
	elseif detector == "ET" || detector == "CE"
		min = 1
		max = 4
	elseif detector == "LISA"
		min = 4
		max = 9
	else
		error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
	end
	file_path = "/home/iara/Desktop/rayleigh/data_all/z100/FisherErrorsRedshift_01.5.SXSBBH0593_"*detector*"_FH_"
	DataModes = ImportModesGrid(file_path, min, max, mode, true)
	

	for num_pars in ["4", "6"]
		NumResolv = Dict()
        close("all")
	    fig, ax1 = subplots()
        
		ax1.set_xscale("log")  
        ax1.set_yscale("log")  
        ax1.set_ylim(1e-2, 1e2)
        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
        ax2.set_ylabel("Luminosity distance [Gpc]")
    
        ax1.set_ylabel("redshift")
		ax1.set_xlabel(L"final mass $[M_\odot]$")
		legends_lines = []
		legends_labels = String[]
		for (k,v) in DataModes
			X, Y, Z = v[num_pars][1], v[num_pars][2], v[num_pars][8]

			ax1.contourf(X,Y, Z, levels=[.9,2], cmap = cmap_f[k], alpha = 0.5)		
			ax1.contour(X,Y, Z, levels = 1, cmap = cmap_f[k], linewidths = 3)	
			
			push!(legends_lines, mpl.lines.Line2D([0], [0], color=cols[k], linewidth=3))
			push!(legends_labels, k)
        end
        ax1.set_xlim(left = 1e1)
        extra = mpl.lines.Line2D([0], [0], color="white")
		legend_extra = legend([extra], [L"$q = 1.5$"], handlelength = 0, fontsize = SMALL_SIZE, frameon = false, 	bbox_to_anchor=(0.09, 0.999))
		ax1.legend(legends_lines,legends_labels, loc = "upper right")
	
		gca().add_artist(legend_extra)
		
		fig.tight_layout()

        ax1.set_title(detector*", "*num_pars*" parameters")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/"*detector*"_"*mode*"_"*num_pars*".pdf")
	end
end


function PlotHorizonGridContour(DataModes, label, num_par="4", parameter = "tau")
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	font_size = 28
	rcParams["xtick.labelsize"]= font_size
	rcParams["ytick.labelsize"]= font_size

	#close("all")
	fig, ax1 = subplots()
	#ax1.set_ylim(1e-1,20 + 1e-20)
	#ax1.set_yticks([0,5,10,15,20])
	#ax1.set_yticklabels([0,5,10,15,20])
	ax1.set_yscale("log")
	ax1.set_xscale("log")
	#ax2 = ax1.twinx()
	#mn, mx = ax1.get_ylim()
	#ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
	#ax2.set_ylabel("Luminosity distance [Gpc]", fontsize = font_size)

	ax1.set_ylabel("redshift", fontsize = font_size)
	ax1.set_xlabel(L"final mass $[M_\odot]$", fontsize = font_size)

	#ax1.set_xlim(4+1e-20,9 + 1e-20)
	#ax1.set_xticks([4,5,6,7,8,9])
	#ax1.set_xticklabels([L"$10^4$", L"$10^5$", L"$10^6$", L"$10^7$", L"$10^8$", L"$10^9$"])
	ax1.tick_params(axis="both", pad=10)
	c_221, c_33, c_44, c_21 = "darkred", "#332288", "#2ca02c", "#ff7f0e"
	c_221_l, c_33_l, c_44_l, c_21_l = "#c47f7f", "#9890c3", "#95cf95", "#ffbe86"
	c_221_l, c_33_l, c_44_l, c_21_l = "tab:red", "tab:blue", "tab:green", "tab:orange"
	cmap_line = Dict(
		"(2,2,0)+(2,2,1)" => mpl.colors.ListedColormap([c_221_l]),
		"(2,2,0)+(3,3,0)" => mpl.colors.ListedColormap([c_33_l]),
		"(2,2,1) I+(3,3,0)" => mpl.colors.ListedColormap([c_33_l]),
		"(2,2,0)+(4,4,0)" => mpl.colors.ListedColormap([c_44_l]),
		"(2,2,1) I+(4,4,0)" => mpl.colors.ListedColormap([c_44_l]),
		"(2,2,0)+(2,1,0)" => mpl.colors.ListedColormap([c_21_l]),
		"(2,2,1) I+(2,1,0)" => mpl.colors.ListedColormap([c_21_l])
		)
	cmap_f = Dict(
		"(2,2,0)+(2,2,1)" => mpl.colors.ListedColormap(["white", c_221_l]),
		"(2,2,0)+(3,3,0)" => mpl.colors.ListedColormap(["white", c_33_l]),
		"(2,2,1) I+(3,3,0)" => mpl.colors.ListedColormap(["white", c_33_l]),
		"(2,2,0)+(4,4,0)" => mpl.colors.ListedColormap(["white", c_44_l]),
		"(2,2,1) I+(4,4,0)" => mpl.colors.ListedColormap(["white", c_44_l]),
		"(2,2,0)+(2,1,0)" => mpl.colors.ListedColormap(["white",c_21_l]),
		"(2,2,1) I+(2,1,0)" => mpl.colors.ListedColormap(["white", c_21_l])
		)
	
	cols = Dict(
		"(2,2,0)+(2,2,1)" => c_221_l,
		"(2,2,0)+(3,3,0)" => c_33_l,
		"(2,2,1) I+(3,3,0)" => c_33_l,
		"(2,2,0)+(4,4,0)" => c_44_l,
		"(2,2,1) I+(4,4,0)" => c_44_l,
		"(2,2,0)+(2,1,0)" => c_21_l,
		"(2,2,1) I+(2,1,0)" => c_21_l
		)
	c_440, c_330, c_221, c_210 = "#00441b", "#08306b", "#67000d", "#3f007d"
	modes = ["(2,2,0)+(2,2,1)", "(2,2,0)+(4,4,0)", "(2,2,0)+(3,3,0)", "(2,2,0)+(2,1,0)"]
	#modes = ["(2,2,1) I+(4,4,0)", "(2,2,1) I+(3,3,0)", "(2,2,0)+(2,2,1) I", "(2,2,1) I+(2,1,0)"]
	#cols = ["#00441b", "#08306b", "#67000d", "#3f007d"]
	legends = []
	legends_labels = String[]
	for k_mode in modes
		v = DataModes[k_mode][num_par]
		if parameter == "Q"
			X, Y, Z_crit, Z_both = v[1], v[2], v[9], v[10]
		else
			X, Y, Z_crit, Z_both = v[1], v[2], v[7], v[8]
		end

		ax1.contourf(X,Y, Z_both, levels=2, cmap = cmap_f[k_mode], alpha = 0.5)		
		ax1.contour(X,Y, Z_both, levels = 1, colors = cols[k_mode], linewidths = 3)	
		a = mpl.lines.Line2D([0], [0], color=cols[k_mode], linewidth=3)
		push!(legends, a)
		push!(legends_labels, k_mode*"oi")
	end
	legend_colors = [cols[i] for i in modes]
	extra = mpl.lines.Line2D([0], [0], color="white")
	legend_extra = legend([extra], [L"$q = 1.5$"], handlelength = 0, fontsize = font_size -10, frameon = false, 	bbox_to_anchor=(0.09, 0.999))
	legend_lines = [mpl.lines.Line2D([0], [0], color=c, linewidth=2) for c in legend_colors]
	#modes = [L"$(2,2,0) + (2,2,1)$", L"$(2,2,0) + (4,4,0)$", L"$(2,2,0) + (3,3,0)$", L"$(2,2,0) + (2,1,0)$"]
	#legend(legend_lines,modes, fontsize = font_size-10, loc = "upper right")
	legend(legends,legends_labels, fontsize = font_size-10, loc = "upper right")
	
	gca().add_artist(legend_extra)
	
	tight_layout()
	fig.suptitle(label*", q = 1.5, "*num_par*" pars", fontsize = font_size)
	savefig(label*"_q15_220_FH_resolboth_"*parameter*"_"*num_par*".pdf")	
end

function ImportDataArrays(file_name, k_mode, k_num)
    data = Dict()

    for i in 1:4
		data["e"*string(i)] = h5open("data_new/"*file_name*"e"*string(i)*".h5") do file
	        read(file) 
	    end
	end

    redshift, M_f = zeros(0), zeros(0)
    df, σf, dtau, σtau = zeros(0), zeros(0), zeros(0), zeros(0)
    dfQ, σfQ, dQ, σQ = zeros(0), zeros(0), zeros(0), zeros(0)


	for (key,value) in data
		for mass in keys(value[k_mode])
			append!(redshift, value[k_mode][string(mass)][k_num][:,1])
			append!(df, value[k_mode][string(mass)][k_num][:,2])
			append!(σf, value[k_mode][string(mass)][k_num][:,3])
			append!(dtau, value[k_mode][string(mass)][k_num][:,4])
			append!(σtau, value[k_mode][string(mass)][k_num][:,5])
			append!(dfQ, value[k_mode][string(mass)][k_num][:,6])
			append!(σfQ, value[k_mode][string(mass)][k_num][:,7])
			append!(dQ, value[k_mode][string(mass)][k_num][:,8])
			append!(σQ, value[k_mode][string(mass)][k_num][:,9])
			for i in 1:length(value[k_mode][string(mass)][k_num][:,1])
				append!(M_f, parse(Float64,mass))
			end	
		end
	end

	X = redshift
	Y = M_f
	Z_ftau, Z_tau, Z_fQ, Z_Q = df./σf, dtau./σtau, dfQ./σfQ, dQ./σQ
	Z_tau_crit, Z_tau_both, Z_Q_crit, Z_Q_both = zeros(0), zeros(0), zeros(0), zeros(0)

	for i in 1:length(Z_ftau)
		if Z_ftau[i] > 1 || Z_tau[i] > 1
			append!(Z_tau_crit,1)
		else
			append!(Z_tau_crit,0)
		end
		if Z_ftau[i] > 1 && Z_tau[i] > 1
			append!(Z_tau_both,1)
		else
			append!(Z_tau_both,0)
		end
		if Z_fQ[i] > 1 || Z_Q[i] > 1
			append!(Z_Q_crit,1)
		else
			append!(Z_Q_crit,0)
		end
		if Z_fQ[i] > 1 && Z_Q[i] > 1
			append!(Z_Q_both,1)
		else
			append!(Z_Q_both,0)
		end
	end
	return X, Y, Z_ftau, Z_tau, Z_fQ, Z_Q, Z_tau_crit, Z_tau_both, Z_Q_crit, Z_Q_both
end

function ImportModesArrays()	
	modes = ["(2,2,0)+(2,2,1) I", "(2,2,0)+(3,3,0)", "(2,2,0)+(4,4,0)", "(2,2,0)+(2,1,0)"]#=,
	"(2,2,1) I+(3,3,0)", "(2,2,1) I+(4,4,0)", "(2,2,1) I+(2,1,0)",
	
	"(3,3,0)+(2,1,0)", "(3,3,0)+(4,4,0)", "(4,4,0)+(2,1,0)", "(2,2,0)+(2,2,1) II", 
	"(2,2,1) II+(3,3,0)", "(2,2,1) II+(4,4,0)", "(2,2,1) II+(2,1,0)"]
	=#
	num_par = ["4"]#,"6"]
	ModesData = Dict()
	for k_mode in modes
		ModesData[k_mode] = Dict()
		for k_num in num_par
			ModesData[k_mode][k_num]  = ImportData("linear_all_", k_mode, k_num)
		end				
	end
	return ModesData	
end

function PlotHorizonBothArrays(DataModes, num_par="4", parameter = "tau")
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	font_size = 28
	rcParams["xtick.labelsize"]= font_size
	rcParams["ytick.labelsize"]= font_size

	close("all")
	fig, ax1 = subplots()
	#ax1.set_ylim(1e-1,20 + 1e-20)
	#ax1.set_yticks([0,5,10,15,20])
	#ax1.set_yticklabels([0,5,10,15,20])
	ax1.set_yscale("log")
	ax1.set_xscale("log")
	ax2 = ax1.twinx()
	mn, mx = ax1.get_ylim()
	ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
	ax2.set_ylabel("Luminosity distance [Gpc]", fontsize = font_size)

	ax1.set_ylabel("redshift", fontsize = font_size)
	ax1.set_xlabel(L"final mass $[M_\odot]$", fontsize = font_size)

	#ax1.set_xlim(4+1e-20,9 + 1e-20)
	#ax1.set_xticks([4,5,6,7,8,9])
	#ax1.set_xticklabels([L"$10^4$", L"$10^5$", L"$10^6$", L"$10^7$", L"$10^8$", L"$10^9$"])
	ax1.tick_params(axis="both", pad=10)
	c_221, c_33, c_44, c_21 = "darkred", "#332288", "#2ca02c", "#ff7f0e"
	c_221_l, c_33_l, c_44_l, c_21_l = "#c47f7f", "#9890c3", "#95cf95", "#ffbe86"
	c_221_l, c_33_l, c_44_l, c_21_l = "tab:red", "tab:blue", "tab:green", "tab:orange"
	cmap = Dict(
		"(2,2,0)+(2,2,1) I" => mpl.colors.ListedColormap([c_221_l]),
		"(2,2,0)+(3,3,0)" => mpl.colors.ListedColormap([c_33_l]),
		"(2,2,1) I+(3,3,0)" => mpl.colors.ListedColormap([c_33_l]),
		"(2,2,0)+(4,4,0)" => mpl.colors.ListedColormap([c_44_l]),
		"(2,2,1) I+(4,4,0)" => mpl.colors.ListedColormap([c_44_l]),
		"(2,2,0)+(2,1,0)" => mpl.colors.ListedColormap([c_21_l]),
		"(2,2,1) I+(2,1,0)" => mpl.colors.ListedColormap([c_21_l])
		)

	cols = Dict(
		"(2,2,0)+(2,2,1) I" => c_221,
		"(2,2,0)+(3,3,0)" => c_33,
		"(2,2,1) I+(3,3,0)" => c_33,
		"(2,2,0)+(4,4,0)" => c_44,
		"(2,2,1) I+(4,4,0)" => c_44,
		"(2,2,0)+(2,1,0)" => c_21,
		"(2,2,1) I+(2,1,0)" => c_21
		)
	c_440, c_330, c_221, c_210 = "#00441b", "#08306b", "#67000d", "#3f007d"
	modes = ["(2,2,0)+(2,2,1) I", "(2,2,0)+(4,4,0)", "(2,2,0)+(3,3,0)", "(2,2,0)+(2,1,0)"]
	cols = [c_221_l, c_44_l, c_33_l, c_21_l]
	modes = ["(2,2,1) I+(4,4,0)", "(2,2,1) I+(3,3,0)", "(2,2,0)+(2,2,1) I", "(2,2,1) I+(2,1,0)"]
	#cols = ["#00441b", "#08306b", "#67000d", "#3f007d"]
	for k_mode in keys(DataModes)
		v = DataModes[k_mode][num_par]
		if parameter == "Q"
			X, Y, Z_crit, Z_both = v[1], log10.(v[2]), v[9], v[10]
		else
			X, Y, Z_crit, Z_both = v[1], (v[2]), v[7], v[8]
		end

		ax1.tricontourf(Y,X, Z_both, levels = 1, cmap = cmap[k_mode], alpha = 0.5)		
		ax1.tricontour(Y,X, Z_both, levels = 1, cmap = cmap[k_mode], linewidths = 2)	
	end
	extra = mpl.lines.Line2D([0], [0], color="white")
	lines = [mpl.lines.Line2D([0], [0], color=c, linewidth=2) for c in cols]
	modes = [L"$(2,2,0) + (2,2,1)$", L"$(2,2,0) + (4,4,0)$", L"$(2,2,0) + (3,3,0)$", L"$(2,2,0) + (2,1,0)$"]

	legend1 = legend([extra], [L"$q = 1.5$"], handlelength = 0, fontsize = font_size -10, frameon = false, 
	bbox_to_anchor=(0.09, 0.999))
	
	legend(lines ,modes, fontsize = font_size-10, loc = "upper right")
	gca().add_artist(legend1)

	tight_layout()
	fig.suptitle("q = 1.5, "*num_par*" pars", fontsize = font_size)
	savefig("LIGO_q15_221_FH_resolboth_"*parameter*"_"*num_par*".pdf")	
end
