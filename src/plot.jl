using PyPlot, HDF5
using PyCall
@pyimport scipy.interpolate as si
mpl = pyimport("matplotlib")

function PlotRayleigh()
    teste = h5open("data/z_delta_sigmaall.h5") do file
        read(file) 
    end
    k_221 = "(2,2,1) I"
    #k_221 = "(3,3,0)"
    k_mass= string(1e8)
    k_num = "4"
    p_221 = teste[k_221][k_mass][k_num]
    masses = zeros(0)
    for k in keys(teste[k_221])
        append!(masses,parse(Float64,k))
    end
    masses = sort(masses)
    #masses = [100000.0, 500000.0, 1.0e6, 5.0e6, 1.0e7, 5.0e7, 1.0e8, 5.0e8, 1.0e9]
    X, Y, df, σf = zeros(0), zeros(0), zeros(0), zeros(0)  
    for mass in masses
        append!(X, teste[k_221][string(mass)][k_num][1,:])
        append!(df, teste[k_221][string(mass)][k_num][2,:])
        append!(σf, teste[k_221][string(mass)][k_num][3,:])

        for i in 1:length(teste[k_221][string(mass)][k_num][1,:])
            append!(Y, mass)
        end
    end
    Z = zeros(0)
    for value in df./σf
    if value > 1
        append!(Z, 1)
    else
        append!(Z,0)
    end
    end
    Z = df./σf
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["mathtext.fontset"] = "cm"
    rcParams["font.family"] = "STIXGeneral"
    rcParams["figure.figsize"] = [15, 8]  # plot image size

    font_size = 28
    rcParams["xtick.labelsize"]= font_size
    rcParams["ytick.labelsize"]= font_size
    lw = 3
    ss = 80
	orig_cmap = mpl.cm.coolwarm
    close("all")
    divnorm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=10)
    divnorm = mpl.colors.LogNorm()#vmin=minimum(Z), vmax=maximum(Z))
    fig1, ax1 = subplots()
    ax1.set_yscale("log")
    tcf = ax1.scatter(X,Y, c= Z, norm = divnorm, cmap="RdBu_r")#orig_cmap)
    cbar = fig1.colorbar(tcf)#, ticks=[minimum(Z) ,1e0,maximum(Z) + 1])
    cbar.set_label("df/σf", rotation=270, fontsize = font_size, labelpad=font_size)
    #ylim(0,1)
    #scatter(teste[k_221][string(k_mass)][k_num][1,:], teste[k_221][string(k_mass)][k_num][2,:]./teste[k_221][string(k_mass)][k_num][3,:])
end
PlotRayleigh()