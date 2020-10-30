using DelimitedFiles, Distributed, HDF5, Random, Distributions, ProgressMeter, Dierckx, PyPlot
include("AllFunctions.jl")
include("Constants.jl")
using .FrequencyTransforms, .PhysConstants, .Quantities

import Base.Threads.@spawn

function ComputeSNR1mode(M_f, mass_f, F_Re, F_Im, mode_1, noise, amplitudes, phases, omega, redshift, convention = "FH")
    # Source parameters
    M_final = (1+redshift)*M_f
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)

    time_unit = (M_total)*PhysConstants.tSun
    strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)

    freq = noise["freq"]
    if convention == "FH" || convention == "EF"
        ft_Re = time_unit*strain_unit*abs.(FrequencyTransforms.Fourier_1mode.("real", freq*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], convention))
        ft_Im = time_unit*strain_unit*abs.(FrequencyTransforms.Fourier_1mode.("imaginary", freq*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], convention))
        SNR = Quantities.SNR_QNM(noise, ft_Re, ft_Im, F_Re, F_Im)
    else
        error("convention argument must be set to \"FH\" or \"EF\".")
    end
    return SNR
end

function ComputeSNRApprox1(M_f, mass_f, mode, noise, amplitudes, omega, redshift)
    # Source parameters
    M_final = (1+redshift)*M_f
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)

    time_unit = (M_total)*PhysConstants.tSun
    strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)

    freq, tau, Q_factor = Dict(), Dict(), Dict()

    # Remnant BH frequency and decay time

    freq[mode] = omega[mode][1]/2/pi/time_unit
    tau[mode] =  time_unit/omega[mode][2]
    Q_factor[mode] = π*freq[mode]*tau[mode]

    noise_f = noise(freq[mode])
    if noise_f == 0
        noise_f = 1e3
    end

    SNR = strain_unit^2*Q_factor[mode]*amplitudes[mode]^2/(20*π^2*freq[mode]*noise_f^2)
    return SNR         
end

function ComputeSNRApprox2(M_f, mass_f, mode_1, mode_2, noise, amplitudes, phases, omega, redshift)
    # Source parameters
    M_final = (1+redshift)*M_f
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)

    time_unit = (M_total)*PhysConstants.tSun
    strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)

    freq, tau, Q_factor = Dict(), Dict(), Dict()

    # Remnant BH frequency and decay time
    for (key, value) in omega
        freq[key] = value[1]/2/pi/time_unit
        tau[key] =  time_unit/value[2]
        Q_factor[key] = π*freq[key]*tau[key]
    end
    noise_1 = noise(freq[mode_1])
    noise_2 = noise(freq[mode_2])
    if noise_1 == 0
        noise_1 = 1e3
    end
    if noise_2 == 0
        noise_2 = 1e3
    end
    
    SNR_1 = (strain_unit*amplitudes[mode_1])^2*Q_factor[mode_1]^3/
            (5*π^2*freq[mode_1]*(1 + 4*Q_factor[mode_1]^2)*noise_1^2)
    SNR_2 = (strain_unit*amplitudes[mode_2])^2*Q_factor[mode_2]*(sin(phases[mode_2]- phases[mode_1])^2 + 2*Q_factor[mode_2]^2)/
            (10*π^2*freq[mode_2]*(1 + 4*Q_factor[mode_2]^2)*noise_2^2)
    if mode_1[2] == mode_2[2] && mode_1[4] == mode_2[4] 
        SNR = sqrt(SNR_1 + SNR_2)
    else 
        Λp = freq[mode_2]^2*Q_factor[mode_1]^2 + 2*freq[mode_1]*freq[mode_2]*Q_factor[mode1]*Q_factor[mode_2] +
            Q_factor[mode_2]*(freq[mode_1]^2 + 4*(freq[mode_1] + freq[mode_2])^2*Q_factor[mode_1]^2)
        Λm = freq[mode_2]^2*Q_factor[mode_1]^2 + 2*freq[mode_1]*freq[mode_2]*Q_factor[mode1]*Q_factor[mode_2] +
            Q_factor[mode_2]*(freq[mode_1]^2 + 4*(freq[mode_1] - freq[mode_2])^2*Q_factor[mode_1]^2)
        extra = strain_unit^2*amplitudes[mode_1]*amplitudes[mode_2]/(5*π*noise_1^2)*
                (16*freq[mode_1]*freq[mode_2]*Q_factor[mode_1]^3*Q_factor[mode_2]^3*
                (freq[mode_1]*Q_factor[mode2] + freq[mode_2]*Q_factor[mode_1])*cos(phases[mode_2]- phases[mode_1]))/Λp/Λm
        SNR = sqrt(SNR_1 + SNR_2 + extra)
    end
    return SNR         
end


function ComputeSNR(M_f, mass_f, F_Re, F_Im, mode_1, mode_2, amplitudes, phases, omega, redshift, detector, convention = "FH")
    # Source parameters
    M_final = (1+redshift)*M_f
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)

    time_unit = (M_total)*PhysConstants.tSun
    strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)

    SNR = Dict()
    
    
    if convention == "FH" || convention == "EF"
        noise = ImportDetectorStrain(detector, false)
        ft_Re, ft_Im =  Fourier_2QNMs(noise["freq"], amplitudes, phases, omega, mode_1, mode_2, time_unit, strain_unit, convention)
        SNR[mode_1] = Quantities.SNR_QNM(noise["freq"], noise["psd"], ft_Re[mode_1], ft_Im[mode_1], F_Re, F_Im)
        SNR[mode_2] = Quantities.SNR_QNM(noise["freq"], noise["psd"], ft_Re[mode_2], ft_Im[mode_2], F_Re, F_Im)
        SNR[mode_1*" + "*mode_2]  = Quantities.SNR_QNM(noise["freq"], noise["psd"], ft_Re[mode_1*" + "*mode_2], ft_Im[mode_1*" + "*mode_2], F_Re, F_Im)
        #loglog(noise["freq"], noise["psd"])
        #loglog(noise["freq"], abs.(ft_Re[mode_1]))
    elseif convention == "DFT"
        noise_itp = ImportDetectorStrain(detector, true)
        ft_Re, ft_Im, ft_freq =  Fourier_2QNMs(0, amplitudes, phases, omega, mode_1, mode_2, time_unit, strain_unit, convention)
        noise_values = noise_itp(ft_freq[mode_1])
        for i in 1:length(noise_values)
            if noise_values[i] == 0
                noise_values[i] = 1e3
            end
        end
        SNR[mode_1] = Quantities.SNR_QNM(ft_freq[mode_1], noise_values, ft_Re[mode_1], ft_Im[mode_1], F_Re, F_Im)
        SNR[mode_2] = Quantities.SNR_QNM(ft_freq[mode_2], noise_values, ft_Re[mode_2], ft_Im[mode_2], F_Re, F_Im)
        SNR[mode_1*" + "*mode_2]  = Quantities.SNR_QNM(ft_freq[mode_1], noise_values, ft_Re[mode_1*" + "*mode_2], ft_Im[mode_1*" + "*mode_2], F_Re, F_Im)
        #loglog(ft_freq[mode_1], noise_values)
        #loglog(ft_freq[mode_1], abs.(ft_Re[mode_1]), ls = "--")
    elseif convention == "Approx"
        noise_itp = ImportDetectorStrain(detector, true)
        SNR[mode_1] = ComputeSNRApprox1(M_f, mass_f, mode_1, noise_itp, amplitudes, omega, redshift)
        SNR[mode_2] = ComputeSNRApprox1(M_f, mass_f, mode_2, noise_itp, amplitudes, omega, redshift)
        SNR[mode_1*" + "*mode_2]  = ComputeSNRApprox2(M_f, mass_f, mode_1, mode_2, noise_itp, amplitudes, phases, omega, redshift)
    else
        error("convention argument must be set to \"FH\", \"EF\" or \"DFT\".")
    end
    return SNR
end

function RunAllSXSFolders(masses, detector, F_Re, F_Im, N_avg, label, convention = "FH")
	noise = ImportDetectorStrain(detector, true)
	folders = readdir("../q_change/")
    for simu_folder_name in folders
        if occursin("10.0", simu_folder_name)
            println(simu_folder_name)
            simu_folder = "../q_change/"*simu_folder_name
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
            @everywhere include("src/RayleighCriterion.jl")
            mode_1 = "(2,2,0)"
            modes = ["(2,2,1) I"]#, "(3,3,0)", "(4,4,0)", "(2,1,0)"]
            amplitudes["(2,2,1) I"] = 0
            N = Int(2e2)
            z_min, z_max = 1e-3, 20
            z_range = range(z_min, stop = z_max, length = N)
            file_path = "data/z_SNR_"*string(label)*".h5"
			if isfile(file_path)
				rm(file_path)
			end
            save_dict = Dict()
            
            for mode_2 in modes
                @showprogress for M_f in masses
                    # choose random redshift
                    z_rand = (rand(Uniform(z_min, z_max)))

                    # save to dictionary 
                    save_dir = mode_1*"+"*mode_2*"/"*string(M_f)
                    save_dict[save_dir] = zeros(N, 2)
                    Base.Threads.@threads for z_rand in z_range
                        i = findall(x -> x == z_rand, z_range)[1]
                        SNR = ComputeSNR(M_f, mass_f, F_Re, F_Im, mode_1, mode_2, noise, amplitudes, phases, omega, z_rand, convention)
                        save_results =  [z_rand, SNR]
                        for j in 1:length(save_results)
                            save_dict[save_dir][i,j] = save_results[j]
                        end
                    end
                    h5open(file_path, "cw") do file
                        write(file, save_dir, save_dict[save_dir])
                    end
                end
            end
        end
    end
end

function RunAllSXSFolders1(masses, detector, F_Re, F_Im, label, convention = "FH")
	folders = readdir("../q_change/")
    for simu_folder_name in folders
        if occursin("10.0", simu_folder_name)
            println(simu_folder_name)
            simu_folder = "../q_change/"*simu_folder_name
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
            @everywhere include("src/RayleighCriterion.jl")
            mode_1 = "(2,2,0)"
            modes = ["(2,2,1) I"]#, "(3,3,0)", "(4,4,0)", "(2,1,0)"]
            amplitudes["(2,2,1) I"] = 0
            N = Int(2)
            z_min, z_max = 1e-10, 20
            z_range = exp10.(range(log10(z_min), stop = log10(z_max), length = N))
            file_path = "data/SNRFinalMassRedshift_"*detector*"_"*string(label)*"_"*convention*".h5"
			if isfile(file_path)
				rm(file_path)
			end
            save_dict_1, save_dict_2 = Dict(), Dict()
            
            for mode_2 in modes
                @showprogress for M_f in masses
                    # choose random redshift
                    z_rand = (rand(Uniform(z_min, z_max)))
                    # save to dictionary 
                    save_dir_1 = mode_1*"/"*string(M_f)
                    save_dir_2 = mode_1*"+"*mode_2*"/"*string(M_f)
                    save_dict_1[save_dir_1] = zeros(N, 2)
                    save_dict_2[save_dir_2] = zeros(N, 2)
                    Base.Threads.@threads for z_rand in z_range
                        i = findall(x -> x == z_rand, z_range)[1]
                        SNR = ComputeSNR(M_f, mass_f, F_Re, F_Im, mode_1, mode_2, amplitudes, phases, omega, z_rand, detector, convention)
                        SNR_1 = SNR[mode_1]
                        SNR_2 = SNR[mode_1*" + "*mode_2] 
                        save_results_1 =  [z_rand, SNR_1]
                        save_results_2 =  [z_rand, SNR_2]
                        for j in 1:length(save_results_1)
                            save_dict_1[save_dir_1][i,j] = save_results_1[j]
                            save_dict_2[save_dir_2][i,j] = save_results_2[j]
                        end
                    end
                    h5open(file_path, "cw") do file
                        write(file, save_dir_1, save_dict_1[save_dir_1])
                        write(file, save_dir_2, save_dict_2[save_dir_2])
                    end
                end
            end
        end
    end
end

function RunAllSXSFoldersApprox(masses, detector, N_avg, label)
	noise = ImportDetectorStrain(detector)[2]
	folders = readdir("../q_change/")
    for simu_folder_name in folders
        if occursin("10.0", simu_folder_name)
            println(simu_folder_name)
            simu_folder = "../q_change/"*simu_folder_name
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
            @everywhere include("src/RayleighCriterion.jl")
            mode_1 = "(2,2,0)"
            modes = ["(2,2,1) I"]#, "(3,3,0)", "(4,4,0)", "(2,1,0)"]
            amplitudes["(2,2,1) I"] = 0
            N = Int(2e2)
            z_min, z_max = 1e-3, 20
            z_range = exp10.(range(-2, stop = 1, length = N))
            file_path = "data/z_SNR_"*string(label)*".h5"
			if isfile(file_path)
				rm(file_path)
			end
            save_dict_1, save_dict_2 = Dict(), Dict()
            
            for mode_2 in modes
                @showprogress for M_f in masses
                    # choose random redshift
                    z_rand = (rand(Uniform(z_min, z_max)))
                    # save to dictionary 
                    save_dir_1 = mode_1*"/"*string(M_f)
                    save_dir_2 = mode_1*"+"*mode_2*"/"*string(M_f)
                    save_dict_1[save_dir_1] = zeros(N, 2)
                    save_dict_2[save_dir_2] = zeros(N, 2)
                    Base.Threads.@threads for z_rand in z_range
                        i = findall(x -> x == z_rand, z_range)[1]
                        SNR_1 = ComputeSNRApprox1(M_f, mass_f, mode_1, noise, amplitudes, omega, z_rand)
                        SNR_2 = ComputeSNRApprox2(M_f, mass_f, mode_1, mode_2, noise, amplitudes, phases, omega, z_rand)
                        save_results_1 =  [z_rand, SNR_1]
                        save_results_2 =  [z_rand, SNR_2]
                        for j in 1:length(save_results_1)
                            save_dict_1[save_dir_1][i,j] = save_results_1[j]
                            save_dict_2[save_dir_2][i,j] = save_results_2[j]
                        end
                    end
                    h5open(file_path, "cw") do file
                        write(file, save_dir_1, save_dict_1[save_dir_1])
                        write(file, save_dir_2, save_dict_2[save_dir_2])
                    end
                end
            end
        end
    end
end

