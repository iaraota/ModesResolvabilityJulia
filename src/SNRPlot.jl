using PyPlot, HDF5, DelimitedFiles, Interpolations, DataFrames, LaTeXStrings, QuadGK
using PyCall
@pyimport scipy.interpolate as si
mpl = pyimport("matplotlib")
np = pyimport("numpy")

function ImportData(file_name, k_mode)
    data = Dict()
	#=
    for i in 5:7
		data["e"*string(i)] = h5open("data/"*file_name*"e"*string(i)*".h5") do file
	        read(file) 
	    end
	end
	=#
	data["all"] = h5open("data/"*file_name*".h5") do file
		read(file) 
	end
    redshift, M_f, SNR = zeros(0), zeros(0), zeros(0)


	for (key,value) in data
		for mass in keys(value[k_mode])
			append!(redshift, value[k_mode][string(mass)][:,1])
			append!(SNR, value[k_mode][string(mass)][:,2])
			for i in 1:length(value[k_mode][string(mass)][:,1])
				append!(M_f, parse(Float64,mass))
			end	
		end
	end

	X = redshift
	Y = M_f
	Z = SNR

	return X, Y, Z
end

function AllModesData()	
	modes = ["(2,2,0)", "(2,2,0)+(2,2,1) I"]#, "(2,2,0)+(3,3,0)", "(2,2,0)+(4,4,0)", "(2,2,0)+(2,1,0)"]#=,
	#="(2,2,1) I+(3,3,0)", "(2,2,1) I+(4,4,0)", "(2,2,1) I+(2,1,0)",
	
	"(3,3,0)+(2,1,0)", "(3,3,0)+(4,4,0)", "(4,4,0)+(2,1,0)", "(2,2,0)+(2,2,1) II", 
	"(2,2,1) II+(3,3,0)", "(2,2,1) II+(4,4,0)", "(2,2,1) II+(2,1,0)"]
	=#
	ModesData = Dict()

	for k_mode in modes
		ModesData[k_mode] = ImportData("z_SNR_LIGO_e1-e5_FH", k_mode)
		#ModesData = ImportData("z_SNR_e345_FH", k_mode)
	end				

	return ModesData	
end
function  Plot_q_a()
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	font_size = 28
	rcParams["xtick.labelsize"]= font_size
	rcParams["ytick.labelsize"]= font_size

	close("all")
	fig, ax1 = subplots()
	q = 1:0.5:10
	a_M = [0.6861, 0.6641, 0.6234, 0.5807, 0.5406, 0.5044, 0.4716, 0.4426, 0.4166, 0.3934, 0.3725, 0.3536, 0.3365,
	0.321, 0.3067, 0.2937, 0.2819, 0.2708, 0.2605]
	ylabel(L"$a/M$", fontsize = font_size)
	xlabel(L"q", fontsize = font_size)
	plot(q, a_M)
end

function PlotSNRRedshift(X,Y,Z)
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

	font_size = 28
	rcParams["xtick.labelsize"]= font_size
	rcParams["ytick.labelsize"]= font_size

	close("all")
	fig, ax1 = subplots()
	#=ax1.set_ylim(1e-1,20 + 1e-20)
	ax1.set_yticks([0,5,10,15,20])
	ax1.set_yticklabels([0,5,10,15,20])

	ax2 = ax1.twinx()
	mn, mx = ax1.get_ylim()
	ax2.set_ylim(luminosity_distance(mn)*1e-3, luminosity_distance(mx)*1e-3)
	ax2.set_ylabel("Luminosity distance [Gpc]", fontsize = font_size)

	ax1.set_ylabel("redshift", fontsize = font_size)
	ax1.set_xlabel(L"final mass $[M_\odot]$", fontsize = font_size)

	ax1.set_xlim(4+1e-20,9 + 1e-20)
	ax1.set_xticks([4,5,6,7,8,9])
	ax1.set_xticklabels([L"$10^4$", L"$10^5$", L"$10^6$", L"$10^7$", L"$10^8$", L"$10^9$"])
	=#
	ax1.set_xscale("log")
	#tcf1 = ax1.scatter(X,Y, c= Z, cmap="RdBu_r", norm = mpl.colors.LogNorm(vmin = 1, vmax = 3e3), marker = ",", s = 2, rasterized=true)		
	#tcf1 = ax1.scatter(X,Y, c= Z, cmap="tab20c", norm = mpl.colors.LogNorm(vmin = 1, vmax = 3e3), marker = ",", s = 2, rasterized=true)		
	tcf1 = ax1.scatter(X,Y, c = Z, cmap="tab20c", norm = mpl.colors.LogNorm(vmin = 1e-1, vmax = 3e3), marker = ",", s = 40, rasterized=true)		
	ax1.set_yscale("log")
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
	cbar1 = fig.colorbar(tcf1, cax=cbar_ax)
	cbar1.ax.set_title(L"SNR", fontsize = font_size)
	#ax1.set_ylim(0,0.5)

	#fig.suptitle(k_mode*" - "*k_num*" par", fontsize = font_size)
	fig.suptitle("(2,2,0) - q = 10 - FH", fontsize = font_size)
	savefig("figs/FH_q10_220_z_SNR_ET_new.pdf")	

end

function luminosity_distance(redshift)
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

	E(z) = sqrt(Ω_M*(1+z)^3 + Ω_K*(1+z)^2 + Ω_Λ)
	D_C = D_H*quadgk(x -> 1/E(x), 0, redshift, rtol=1e-18)[1]
	D_L = (1 + redshift)*D_C
	return D_L
end


function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end


function ImportDataGrid(file_name, k_mode,  min, max)	
    files = Dict()

    for i in min:max
		files["e"*string(i)] = h5open("data/"*file_name*".h5") do file
	        read(file, k_mode) 
	    end
	end

	data = reduce(merge, collect(values(files)))
	
    M_f = zeros(0)
    
	for mass in keys(data)
		append!(M_f, parse(Float64,mass))
	end
	M_f = sort(M_f)

	redshift = data[string(M_f[1])][:,1]

	M, N = length(M_f), length(redshift)
	
	SNR = zeros(M, N)

	for i in 1:M
		for j in 1:N
			SNR[i, j] = data[string(M_f[i])][j, 2]
		end
	end

	X, Y = meshgrid(M_f, redshift)

	return X, Y, SNR

end

function ImportModesGrid(file_name, min, max)	
	modes = ["(2,2,0)", "(2,2,0)+(2,2,1) I"]
	#modes = ["(2,2,0)+(2,2,1) I", "(2,2,0)+(3,3,0)", "(2,2,0)+(4,4,0)", "(2,2,0)+(2,1,0)"]
	#"(2,2,1) I+(3,3,0)", "(2,2,1) I+(4,4,0)", "(2,2,1) I+(2,1,0)",
	#"(3,3,0)+(2,1,0)", "(3,3,0)+(4,4,0)", "(4,4,0)+(2,1,0)", "(2,2,0)+(2,2,1) II", 
	#"(2,2,1) II+(3,3,0)", "(2,2,1) II+(4,4,0)", "(2,2,1) II+(2,1,0)"]
	
	#modes = ["(2,2,1) I+(4,4,0)", "(2,2,1) I+(3,3,0)", "(2,2,0)+(2,2,1) I", "(2,2,1) I+(2,1,0)"]

	ModesData = Dict()
	for k_mode in modes
		ModesData[k_mode] = ImportDataGrid(file_name, k_mode, min, max)	
	end
	return ModesData	
end

function PlotHorizonGrid(DataModes,label)
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [12, 8]  # plot image size

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

	modes = ["(2,2,0)"]#, "(2,2,0)+(2,2,1) I"]
	ax1.set_ylim(1e-2, 1e0)
	for k_mode in modes
		v = DataModes[k_mode]
		#cs = ax1.contourf(v[1], v[2], v[3], cmap = "inferno", norm = mpl.colors.LogNorm(vmin = 1e-1, vmax = 3e3))
		cs = ax1.scatter(v[1], v[2], c = v[3],cmap="tab20c", norm = mpl.colors.LogNorm(vmin = 1e-1, vmax = 1e3), marker = ",", s = 40, rasterized=true)		
		fig.colorbar(cs)
	end
	title(label*", q = 10", fontsize = font_size)
	tight_layout()
	savefig("figs/"*label*"_SNR_q1.5.pdf")	
end
