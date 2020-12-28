# Add modules to load path
if !("./src/module" in LOAD_PATH)
    push!(LOAD_PATH, "./src/module")
end

using PyPlot, HDF5, LaTeXStrings
using CSV, DataFrames

using PyCall
mpl = pyimport("matplotlib")

using Quantities

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function ImportDataGridRayleigh(file_name, k_mode, k_num, min, max)	
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

function ImportModesGridRayleigh(file_name, min, max, choose_modes = "All")
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
			ModesData[k_mode][k_num]  = ImportDataGridRayleigh(file_name, k_mode, k_num, min, max)	
		end				
	end

	return ModesData	
end

function ContourPoints(X,Y,Z)
	# Compute contour points of data where Z assume binary values

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

	# Sort points according to the path
	
	M = hcat(x,y)[sortperm(x, rev = true),:]	# sort x reversed
	x, y = M[:,1], M[:,2]

	## select (x,y) while y is increasing  (x is reversed, decreasing)
	i_right = 1
	i_aux = 1
	aux = 0
	while aux == 0
		if y[i_aux] <= y[i_aux + 1]
			i_right = i_aux
			i_aux += 1
		else
			aux = 1
		end
	end
	
	indices_y = [i_right]
	i_max = 0
	for i in i_right:length(y)
		if y[i] >= y[indices_y[end]] && x[i] < x[indices_y[end]]
			append!(indices_y, i)
		end
	end
	
	x_right = x[1:i_right-1]
	y_right = y[1:i_right-1]
	for i in indices_y
		append!(x_right, x[i])
		append!(y_right, y[i])
	end

	deleteat!(x, indices_y)
	deleteat!(y, indices_y)
	x_left = x[i_right+1:end]
	y_left = y[i_right+1:end]

	M = hcat(x_left,y_left)[sortperm(y_left, rev = true),:]	# sort y_left	
	x_left, y_left = M[:,1], M[:,2]
	
	x = vcat(x_right, x_left)
	y = vcat(y_right, y_left)

	# make curve smooth by deleting "zigzag" and repeted points
	
	indices = [1]
	while indices != []
		indices = Int[]
		for i in 2:length(y)-1
			if y[i] <= y[i-1] && y[i] <= y[i+1] 
				append!(indices, i)
			end
		end
		deleteat!(x, indices)
		deleteat!(y, indices)

	end
	
	return x, y
end

function NumberResolvableModes(DataModes, num_par="4", variable = "tau", criterion = "both")
	# Compute the number of resolvable modes
	higher_modes = ["(2,2,1)", "(3,3,0)", "(4,4,0)", "(2,1,0)"]

	ResolveData = Dict()
	X, Y = 0,0
	for (key, value) in DataModes
		X = value[num_par][1]
		Y = value[num_par][2]
		if variable == "tau"
			if criterion == "both"
				ResolveData[key] = value[num_par][8]
			elseif criterion == "critical"
				ResolveData[key] = value[num_par][7]
			else
				error("criterion must be set to \"both\" or \"critical\"!")
			end
		elseif variable == "Q"
			if criterion == "both"
				ResolveData[key] = value[num_par][10]
			elseif criterion == "critical"
				ResolveData[key] = value[num_par][9]
			else
				error("criterion must be set to \"both\" or \"critical\"!")
			end
		else
			error("variable must be set to \"tau\" or \"Q\".")
		end
	end
	
	# merge (2,2,1) two methods by imposing that both should be resolvable
	ResolveData["(2,2,0)+(2,2,1)"] = (ResolveData["(2,2,0)+(2,2,1) I"] .* ResolveData["(2,2,0)+(2,2,1) II"])
	delete!(ResolveData, "(2,2,0)+(2,2,1) I")
	delete!(ResolveData, "(2,2,0)+(2,2,1) II")
	for k in ["(3,3,0)", "(4,4,0)", "(2,1,0)"]
		ResolveData["(2,2,1)+"*k] = (ResolveData["(2,2,1) I+"*k] .* ResolveData["(2,2,1) II+"*k])
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

function SelectSimulation(q_mass)
	# TODO: add all simulations/ optimize simulation selection
	if q_mass == 1.5
		label = "01.5.SXSBBH0593"
	elseif q_mass == 5.0
		label = "05.0.SXSBBH0107"
	elseif q_mass == 10.0
		label = "10.0.SXSBBH1107"
	end
	return label
end


function NewDataNumberResolvable(detector, q_mass = 1.5)
	# Create new file for modes resolvability and number of resolvable modes
	# This makes ploting faster

	if detector == "LIGO"
		min = 1
		max = 4
		smooth = 1
	elseif detector == "ET" || detector == "CE"
		min = 1
		max = 4
	elseif detector == "LISA"
		min = 3
		max = 9
	else
		error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
	end

	label = SelectSimulation(q_mass)
	file_path = "data/rayleigh/data/FisherErrorsRedshift_"*label*"_"*detector*"_FH_"
	file_path = "/home/iara/Desktop/data_rayleigh/data/FisherErrorsRedshift_"*label*"_"*detector*"_FH_"
	DataModes = ImportModesGridRayleigh(file_path, min, max)
	if ! isdir("data/rayleigh/")
		mkdir("data/rayleigh/")
	end

	for num_pars in ["4", "6"]
		NumResolv = Dict()
		X, Y, NumResolv[2], NumResolv[3], NumResolv[4], NumResolv[5] = NumberResolvableModes(DataModes, num_pars)
		h5open("data/rayleigh/NumberModeHorizon_"*label*".h5", "cw") do file
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
			XX = value[num_pars][1]
			YY = value[num_pars][2]
			ResolveData[key] = value[num_pars][8]
		end
		
		# consider the mean value between (2,2,1) two methods
		ResolveData["(2,2,0)+(2,2,1)"] = ResolveData["(2,2,0)+(2,2,1) I"] .* ResolveData["(2,2,0)+(2,2,1) II"]
		delete!(ResolveData, "(2,2,0)+(2,2,1) I")
		delete!(ResolveData, "(2,2,0)+(2,2,1) II")
		for k in ["(3,3,0)", "(4,4,0)", "(2,1,0)"]
			ResolveData["(2,2,1)+"*k] = ResolveData["(2,2,1) I+"*k] .* ResolveData["(2,2,1) II+"*k]
			delete!(ResolveData, "(2,2,1) I+"*k)
			delete!(ResolveData, "(2,2,1) II+"*k)
		end
		h5open("data/rayleigh/ModeHorizon_"*label*".h5", "cw") do file
			write(file, detector*"/"*num_pars*"/mass", XX)
			write(file, detector*"/"*num_pars*"/redshift", YY)
		end
		for (k,v) in ResolveData
			h5open("data/rayleigh/ModeHorizon_"*label*".h5", "cw") do file
				write(file, detector*"/"*num_pars*"/"*k, v)
			end
		end
	end

end

function NewDataAllDetectorsRayleigh(q_mass)
	label = SelectSimulation(q_mass)

	if isfile("data/rayleigh/NumberModeHorizon_"*label*".h5")
		rm("data/rayleigh/NumberModeHorizon_"*label*".h5")
	end

	if isfile("data/rayleigh/ModeHorizon_"*label*".h5")
		rm("data/rayleigh/ModeHorizon_"*label*".h5")
	end
	for detector in ["LIGO", "ET", "CE", "LISA"]
		NewDataNumberResolvable(detector, q_mass)
	end
end

function PlotResolvableModesContourAll(q_mass)
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [20, 8]  # plot image size

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

	colors = Dict(2=> "black", 3 => "tab:red", 4 => "tab:green", 5 => "tab:blue")

	label = SelectSimulation(q_mass)

	DataNResolv = h5open("data/rayleigh/NumberModeHorizon_"*label*".h5") do file
		read(file) 
	end

	# import GW catalog
	input = "data/catalog_mass_dl_z.csv"
	catalog = DataFrame(CSV.File(input))


	for num_pars in ["4", "6"]
		close("all")
		fig, ax1 = subplots()

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
	
		ax1.set_ylabel(L"redshift, $z$")
		ax1.set_xlabel(L"massa final $[M_\odot]$")
		legends_lines = []
		legends_labels = String[]
		for detector in ["LIGO", "CE", "ET", "LISA"]
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
				if all(x->x == 0,  DataNResolv[detector][num_pars][string(k)*"modes"])
					continue
				else
					x, y = ContourPoints(X, Y, DataNResolv[detector][num_pars][string(k)*"modes"])					
					if detector == "LISA"
						ax1.plot(x,y, label = string(k)*" modes", lw = 3, ls = ls, color = colors[k])
					else
						ax1.plot(x,y, lw = 3, ls = ls, color = colors[k])
					end				
					#ax1.fill_between(x,y, 0, alpha = 0.5, color = colors[k])
				end	
				#ax1.contourf(X,Y,  DataNResolv[detector][num_pars][string(k)*"modes"], levels=[.9,2], alpha = 0.5)		
			end
		end

        ax1.set_xlim(1e1, 1e9)
        
		ax1.legend()
		
        extra = mpl.lines.Line2D([0], [0], color="white")
		legend_extra = legend([extra], [latexstring(L"$q = $", q_mass)], handlelength = 0, fontsize = SMALL_SIZE, frameon = false, 	bbox_to_anchor=(0.08, 0.999))
		gca().add_artist(legend_extra)

		# ax1.set_title("Spectroscopy horizon, "*num_pars*" parameters")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/all_Nmodes_"*num_pars*"_"*label*".pdf")
	end
end


function PlotHorizon2modesContourRayleigh(q_mass, choose_modes)
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
		min = 3
		max = 9
	else
		error("Detector must be \"LIGO\", \"ET\", \"CE\" or \"LISA\".")
	end

	if choose_modes == "(2,2,0)"
		modes = ["(2,2,0)+(2,2,1)", "(2,2,0)+(3,3,0)", "(2,2,0)+(4,4,0)", "(2,2,0)+(2,1,0)",]
	elseif choose_modes == "(2,2,1)"
		modes = ["(2,2,1)+(4,4,0)", "(2,2,1)+(3,3,0)", "(2,2,0)+(2,2,1)", "(2,2,1)+(2,1,0)"]
	elseif choose_modes == "(3,3,0)"
		modes = ["(2,2,0)+(3,3,0)", "(2,2,1)+(3,3,0)", "(3,3,0)+(2,1,0)", "(3,3,0)+(4,4,0)"]
	elseif choose_modes == "(4,4,0)"
		modes = ["(2,2,0)+(4,4,0)", "(2,2,1)+(4,4,0)", "(4,4,0)+(2,1,0)", "(3,3,0)+(4,4,0)"]
	elseif choose_modes == "(2,1,0)"
		modes = ["(2,2,0)+(2,1,0)", "(2,2,1)+(2,1,0)", "(3,3,0)+(2,1,0)", "(4,4,0)+(2,1,0)"]
	end	

	label = SelectSimulation(q_mass)

	DataModes = h5open("data/rayleigh/ModeHorizon_"*label*".h5") do file
		read(file) 
	end

	for num_pars in ["4", "6"]
        close("all")
	    fig, ax1 = subplots()
        
		ax1.set_xscale("log")  
        ax1.set_yscale("log")  
        ax1.set_ylim(1e-2, 1e3)
        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
		ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
        ax2.set_yscale("log")  
		
        ax2.set_ylabel("Luminosity distance [Gpc]")
    
        ax1.set_ylabel("redshift")
		ax1.set_xlabel(L"final mass $[M_\odot]$")

		X, Y = DataModes[detector][num_pars]["mass"], DataModes[detector][num_pars]["redshift"]
		for k_modes in modes
			Z = DataModes[detector][num_pars][k_modes]
			if all(x->x==0, Z)
				continue
			else
				x, y = ContourPoints(X, Y, Z)
				ax1.plot(x,y, label = k_modes, lw = 3, ls = "-")
				ax1.fill_between(x,y, 0, alpha = 0.5)
			end

        end
        ax1.set_xlim(left = 1e1)
        ax1.legend()
	
		
		fig.tight_layout()

        ax1.set_title(detector*", "*num_pars*" parameters, q = $q_mass")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/"*detector*"_"*choose_modes*"_"*num_pars*"_"*label*".pdf")
	end
end


function PlotHorizon2modesContourRayleighAllDetectors(q_mass, choose_modes)
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [20, 8]  # plot image size

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


	if choose_modes == "(2,2,0)"
		modes = ["(2,2,0)+(2,2,1)", "(2,2,0)+(3,3,0)", "(2,2,0)+(4,4,0)", "(2,2,0)+(2,1,0)",]
	elseif choose_modes == "(2,2,1)"
		modes = ["(2,2,1)+(4,4,0)", "(2,2,1)+(3,3,0)", "(2,2,0)+(2,2,1)", "(2,2,1)+(2,1,0)"]
	elseif choose_modes == "(3,3,0)"
		modes = ["(2,2,0)+(3,3,0)", "(2,2,1)+(3,3,0)", "(3,3,0)+(2,1,0)", "(3,3,0)+(4,4,0)"]
	elseif choose_modes == "(4,4,0)"
		modes = ["(2,2,0)+(4,4,0)", "(2,2,1)+(4,4,0)", "(4,4,0)+(2,1,0)", "(3,3,0)+(4,4,0)"]
	elseif choose_modes == "(2,1,0)"
		modes = ["(2,2,0)+(2,1,0)", "(2,2,1)+(2,1,0)", "(3,3,0)+(2,1,0)", "(4,4,0)+(2,1,0)"]
	end	

	label = SelectSimulation(q_mass)

	DataModes = h5open("data/rayleigh/ModeHorizon_"*label*".h5") do file
		read(file) 
	end

	# import GW catalog
	input = "data/catalog_mass_dl_z.csv"
	catalog = DataFrame(CSV.File(input))

	for num_pars in ["4", "6"]
        close("all")
	    fig, ax1 = subplots()

		ax1.set_xscale("log")  
        ax1.set_yscale("log")  
        ax1.set_ylim(1e-2, 1e3)
        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
		ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
        ax2.set_yscale("log")  
		
        ax2.set_ylabel("distância luminosa [Gpc]")
    
        ax1.set_ylabel(L"redshift, $z$")
		ax1.set_xlabel(L"massa final $[M_\odot]$")

		# plot catalog
		ax1.scatter(catalog["final_mass"], catalog["redshift"], marker = "*", s = 150, 
					color = "gold", edgecolors = "darkgoldenrod")
		
		for detector in ["LIGO", "CE", "ET", "LISA"]
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
			colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
			col_i = 1

			X, Y = DataModes[detector][num_pars]["mass"], DataModes[detector][num_pars]["redshift"]
			for k_modes in modes
				Z = DataModes[detector][num_pars][k_modes]
				if all(x->x==0, Z)
					continue
				else
					x, y = ContourPoints(X, Y, Z)
					if detector == "LISA"
						ax1.plot(x,y, label = k_modes, lw = 3, ls = ls, color = colors[col_i])
					else
						ax1.plot(x,y, lw = 3, ls = ls, color = colors[col_i])
					end
					#ax1.fill_between(x,y, 0, alpha = 0.5)
				end
				col_i += 1
			end
		end
        ax1.set_xlim(1e1, 1e9)
        ax1.legend()
	
		
		fig.tight_layout()

        extra = mpl.lines.Line2D([0], [0], color="white")
		legend_extra = legend([extra], [latexstring(L"$q = $", q_mass)], handlelength = 0, fontsize = SMALL_SIZE, frameon = false, 	bbox_to_anchor=(0.08, 0.999))
		gca().add_artist(legend_extra)

		# ax1.set_title("Spectroscopy horizon, "*num_pars*" parameters")

        fig.tight_layout()
        if ! isdir("figs/rayleigh/")
            mkdir("figs/rayleigh/")
        end
		savefig("figs/rayleigh/AllDetectors_"*choose_modes*"_"*num_pars*"_"*label*".pdf")
	end
end

