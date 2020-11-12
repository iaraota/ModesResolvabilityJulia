# Add modules to load path
if !("./src/module" in LOAD_PATH)
    push!(LOAD_PATH, "./src/module")
end

using PyPlot, HDF5, LaTeXStrings

using PyCall
mpl = pyimport("matplotlib")

using Quantities

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function ImportDataGridSNR(file_name, k_mode, k_num, min, max)	
	data = h5open(file_name*".h5") do file
		read(file, k_mode) 
	end

    M_f = zeros(0)
	
	for mass in keys(data)
		append!(M_f, parse(Float64,mass))
	end
	M_f = sort(M_f)
	
	redshift = data[string(M_f[1])][k_num][:,1]

	M, N = length(M_f), length(redshift)

	SNR, SNR_horizon = zeros(M,N), zeros(M,N)

	for i in 1:M
		for j in 1:N
			SNR[i, j] = data[string(M_f[i])][k_num][j, 2]
			if SNR[i, j] >= 8 
				SNR_horizon[i, j] = 1
			else
				SNR_horizon[i,j] = 0
			end
		end
	end

	X, Y = meshgrid(M_f, redshift)
	return X, Y, SNR, SNR_horizon
end

function ImportModesGridSNR(file_name, min, max, merge_221 = false)
	modes = ["(2,2,0)", "(2,2,1) I", "(2,2,1) II", "(3,3,0)", "(4,4,0)", "(2,1,0)"]

	num_par = ["4","6"]
	ModesData = Dict()
	for k_mode in modes
		ModesData[k_mode] = Dict()
		for k_num in num_par
			ModesData[k_mode][k_num]  = ImportDataGridSNR(file_name, k_mode, k_num, min, max)	
		end				
	end

	return ModesData	
end

function ContourPointsSNR(X,Y,Z)
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

function NewDataSNR(detector, q_mass = 1.5, convention = "FH")
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
	file_path = "data/SNR/SNR_"*label*"_"*detector*"_"*convention*"_all_masses"
	DataModes = ImportModesGridSNR(file_path, min, max)


	for num_pars in ["4", "6"]

		X, Y = 0,0
		SNR = Dict()
		SNRHorizon = Dict()
		for (key, value) in DataModes
			# TODO: choose critical or both, tau or Q
			X = value[num_pars][1]
			Y = value[num_pars][2]
			SNR[key] = value[num_pars][3]
			SNRHorizon[key] = value[num_pars][4]
		end
		
		# consider the mean value between (2,2,1) two methods
		SNR["(2,2,1)"] = (SNR["(2,2,1) I"] .+ SNR["(2,2,1) II"])/2
		delete!(SNR, "(2,2,1) I")
		delete!(SNR, "(2,2,1) II")

		SNRHorizon["(2,2,1)"] = (SNRHorizon["(2,2,1) I"] .* SNRHorizon["(2,2,1) II"])
		delete!(SNRHorizon, "(2,2,1) I")
		delete!(SNRHorizon, "(2,2,1) II")

		h5open("data/SNR/SNRMatrix_"*label*"_"*convention*".h5", "cw") do file
			write(file, detector*"/"*num_pars*"/mass", X)
			write(file, detector*"/"*num_pars*"/redshift", Y)
		end
		for (k,v) in SNR
			h5open("data/SNR/SNRMatrix_"*label*"_"*convention*".h5", "cw") do file
				write(file, detector*"/"*num_pars*"/"*k, v)
			end
		end


		h5open("data/SNR/SNRHorizon_"*label*"_"*convention*".h5", "cw") do file
			write(file, detector*"/"*num_pars*"/mass", X)
			write(file, detector*"/"*num_pars*"/redshift", Y)
		end
		for (k,v) in SNRHorizon
			h5open("data/SNR/SNRHorizon_"*label*"_"*convention*".h5", "cw") do file
				write(file, detector*"/"*num_pars*"/"*k, v)
			end
		end
	end

end

function NewDataAllDetectorsSNR(q_mass = 1.5, convention = "FH")
	label = SelectSimulation(q_mass)
	file_path = "data/SNR/SNRMatrix_"*label*"_"*convention*".h5"
	if isfile(file_path)
		rm(file_path)
	end
	file_path = "data/SNR/SNRHorizon_"*label*"_"*convention*".h5"
	if isfile(file_path)
		rm(file_path)
	end
	for detector in ["LIGO", "ET", "CE", "LISA"]
		NewDataSNR(detector, q_mass, convention)
	end
end

function PlotSNRHorizonAll(q_mass, convention = "FH")
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

	colors = Dict("(2,2,0)"=> "black", "(2,2,1)" => "tab:red", "(4,4,0)" => "tab:green", "(3,3,0)" => "tab:blue", "(2,1,0)" => "tab:orange")
	cmaps = Dict("(2,2,0)"=> "Greys", "(2,2,1)" => "Reds", "(4,4,0)" => "Greens", "(3,3,0)" => "Blues", "(2,1,0)" => "Oranges")

	label = SelectSimulation(q_mass)

	SNRHorizon = h5open("data/SNR/SNRHorizon_"*label*"_"*convention*".h5") do file
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
		ax2.set_yscale("log")  
		ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
		ax2.set_ylabel("Luminosity distance [Gpc]")
	
		ax1.set_ylabel("redshift")
		ax1.set_xlabel(L"final mass $[M_\odot]$")
		legends_lines = []
		legends_labels = String[]
		for detector in [s"LIGO", "CE", "ET", "LISA"]
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
			X, Y = SNRHorizon[detector][num_pars]["mass"], SNRHorizon[detector][num_pars]["redshift"]
			#X = (1 .+ Y).*X

			for k in ["(2,2,0)", "(2,2,1)", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
				if all(x->x == 0,  SNRHorizon[detector][num_pars][k])
					continue
				else
					x, y = ContourPointsSNR(X, Y, SNRHorizon[detector][num_pars][k])					
					if detector == "LISA"
						ax1.plot(x,y, label = k, lw = 3, ls = ls, color = colors[k])
					else
						ax1.plot(x,y, lw = 3, ls = ls, color = colors[k])
					end				
					#ax1.fill_between(x,y, 0, alpha = 0.5, color = colors[k])
				end	
				#ax1.contourf(X,Y,  SNRHorizon[detector][num_pars][k], levels=[.9,2], alpha = 0.5, cmap = cmaps[k])		
				#ax1.scatter(X,Y, c = SNRHorizon[detector][num_pars][k], alpha = 0.2, cmap = cmaps[k], rasterized = true)		
			end
		end

        ax1.set_xlim(1e1, 1e9)
        
		ax1.legend()
		
        extra = mpl.lines.Line2D([0], [0], color="white")
		legend_extra = legend([extra], [latexstring(L"$q = $", q_mass)], handlelength = 0, fontsize = SMALL_SIZE, frameon = false, 	bbox_to_anchor=(0.09, 0.999))
		gca().add_artist(legend_extra)

		ax1.set_title("SNR horizon, "*num_pars*" parameters, "*convention)

        fig.tight_layout()
        if ! isdir("figs/SNR/")
            mkdir("figs/SNR/")
        end
		savefig("figs/SNR/SNRHorizon_"*num_pars*"_"*label*"_"*convention*".pdf")
	end
end
