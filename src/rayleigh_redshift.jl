
using HDF5 # load HDF5 files
using DelimitedFiles # load txt files to arrays
using Distributed
using PyPlot
using LaTeXStrings

import Base.Threads.@spawn

function integral_trapezio(f, x)
    y = 0
    @fastmath @inbounds @simd for i in 2:length(f)
        y += (x[i] - x[i-1])*(f[i] + f[i-1])/2
    end
    return y
end

include("fourier_derivs.jl") # import fourier transform of partial derivatives of the waveform
function resolvability(masses, detector, F_Re, F_Im, N_avg, label)
    ####################################################################################################################
    # import PSD noise
    ## LISA Strain
    LISA_strain = readdlm("LISA_Strain_Sensitivity_range.txt", comments=true, comment_char='#')
    ## Design sensitivity
    aLIGO_strain = readdlm("aLIGODesign.txt", comments=true, comment_char='#')
    ## O1-H1 near GW150914
    #H1O1_strain = readdlm("import_data/H1-GDS-CALIB_STRAIN.txt", comments=true, comment_char='#')

    # choose noise
    noise = Dict()
    if detector == "LIGO"
        noise["freq"], noise["psd"] = aLIGO_strain[:,1], aLIGO_strain[:,2]
        noise_name = "Design sensitivity"
        # limit maximum frequency
        noise["psd"] = noise["psd"][(noise["freq"] .< 5000)]
        noise["freq"] = noise["freq"][(noise["freq"] .< 5000)]

    elseif detector == "LISA"
        noise["freq"], noise["psd"] = LISA_strain[:,1], LISA_strain[:,2]
        noise_name = "LISA sensitivity"
    else
        return print("Wrong detector option! Choose LIGO or LISA")
    end
    ########################################################################################################################
    # constants
    Gconst = 6.67259e-11  # m^3/(kg*s^2)
    clight = 2.99792458e8  # m/s
    MSol = 1.989e30  # kg
    parsec = 3.08568025e16  # m

    # in seconds
    tSol = MSol*Gconst / clight^3  # from Solar mass to seconds
    Dist = 1.0e6*parsec / clight  # from Mpc to seconds
    ####################################################################################################################
    save_SNR, save_max_dist, save_redshift = Dict(), Dict(), Dict()
    folders = readdir()
    for simu_folder in folders
        if occursin("SXS", simu_folder)
            println(simu_folder)
            ## fitted parameters
            ratios = h5open(simu_folder*"/arrays/fits/ratios.h5", "r") do file
                read(file)
            end
            amplitudes = h5open(simu_folder*"/arrays/fits/amplitudes.h5", "r") do file
                read(file)
            end
            phases = h5open(simu_folder*"/arrays/fits/phases.h5", "r") do file
                read(file)
            end
            omega = h5open(simu_folder*"/arrays/fits/omega.h5", "r") do file
                read(file)
            end
            dphases = Dict()
            for (key,value) in phases
                dphases[key] = phases["(2,2,0)"] - value
            end
            # Consider number of expected detection in (3,3,0) mode
            for (key, value) in amplitudes
                amplitudes[key] = abs(amplitudes[key])
            end
            amplitudes["(3,3,0)"] = amplitudes["(3,3,0)"]/N_avg
            ratios["(3,3,0)"] = ratios["(3,3,0)"]/N_avg
            # final mass
            mass_f = 0.0
            open(simu_folder*"/import_data/metadata.txt") do file
                parameters = Dict()
                for line in eachline(file)
                    # final mass
                    if occursin("remnant-mass", line)
                        mass_f = parse(Float64, split(line)[3])
                    end
                end
            end
            ####################################################################################################################
            @everywhere include("rayleigh_all_z_calc.jl")
            mode_1 = "(2,2,0)"
            modes = ["(2,2,1) I", "(2,2,1) II", "(3,3,0)"]
            nums = [4, 6]
            file_name = string(detector)*"_"*string(F_Re)*"+"*string(F_Im)*"_N"*string(N_avg)*"_"*string(label)*"_"*string(simu_folder)
            Base.Threads.@threads for mode_2 in modes
                for M_f in masses
                    for num_par in nums
                        rayleight_calc(M_f, mass_f, F_Re, F_Im, num_par, mode_1, mode_2, noise, amplitudes, phases, ratios, dphases, omega, file_name)
                    end
                end
            end
        end
    end

end

println("To compute the SNR min use: ")
println("resolvability(masses, detector, F_Re, F_Im, N_avg, label)")
println("Where masses can be and array of several values, detector is \"LIGO\" or \"LISA\", F_Re and F_Im are responses for each polarization, N_avg is the expected number of dections weight in the (3,3,0) amplitude and label is the saved file label")
#resolvability([1e+5, 1e+6, 1e+7, 1e+8], 3e+3, "LISA", 1, 0)
#resolvability(63, 430, mass_f, "LIGO", 1, 1im)
