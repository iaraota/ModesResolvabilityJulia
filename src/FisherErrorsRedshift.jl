using DelimitedFiles, Distributed, HDF5, Random, Distributions, ProgressMeter, LinearAlgebra
include("AllFunctions.jl")
include("Constants.jl")
using .FrequencyTransforms, .PhysConstants, .Quantities

import Base.Threads.@spawn

function RunAllSXSFoldersLinearAllModes(masses, detector, F_Re, F_Im, label, simulation_folder = "SXS", convention = "FH")
	noise = ImportDetectorStrain(detector)[1]
	folders = readdir("../q_change/")
    for simu_folder_name in folders
        if occursin(simulation_folder, simu_folder_name)
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
            modes = ["(2,2,1) I", "(2,2,1) II", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
            all_modes = ["(2,2,0)", "(2,2,1) I", "(2,2,1) II", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
            nums = [4, 6]
            z_min = 1e-2
            if detector == "LIGO"
                N = Int(2e2)
                z_max = 5
            else
                N = Int(1e3)
                z_max = 20
            end
            z_range = exp10.(range(log10(z_min), stop = log10(z_max), length = N))
            file_path = "data/FisherErrorsRedshift_"*simu_folder_name*"_"*detector*"_"*string(convention)*"_"*string(label)*".h5"
			if isfile(file_path)
				rm(file_path)
			end
            save_dict = Dict()
            perm_modes = Dict()
            @showprogress for M_f in masses
                perm_modes[M_f] = [0,0]
                for mode_1 in all_modes
                    for mode_2 in all_modes
                        condition_1 = mode_1 != mode_2
                        condition_2 = sort([mode_1, mode_2]) != ["(2,2,1) I", "(2,2,1) II"]
                        condition_3 = all([sort([mode_1,mode_2]) != sort(x) for x in eachcol(perm_modes[M_f])])
                        if condition_1 && condition_2 && condition_3
                            perm_modes[M_f] = hcat(perm_modes[M_f], [mode_1, mode_2])
                            for num_par in nums
                                save_dir = mode_1*"+"*mode_2*"/"*string(M_f)*"/"*string(num_par)
                                save_dict[save_dir] = zeros(N, 9)
                                Base.Threads.@threads for z_rand in z_range
                                    i = findall(x -> x == z_rand, z_range)[1]
                                    sigma, delta_var = RayleighCriterion(M_f, mass_f, F_Re, F_Im, num_par, mode_1, mode_2, noise, amplitudes, phases, ratios, dphases, omega, z_rand, convention)
                                    save_results =  [z_rand, delta_var["f"], maximum(sigma["f_tau"]), delta_var["tau"], maximum(sigma["tau"]), delta_var["f"], maximum(sigma["f_Q"]), delta_var["Q"], maximum(sigma["Q"])]
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
        end
    end
end


function RunAllSXSFoldersLinearChooseModes(masses, detector, F_Re, F_Im, label, mode_1, mode_2, simulation_folder = "SXS", convention = "FH")
	noise = ImportDetectorStrain(detector)[1]
	folders = readdir("../q_change/")
    for simu_folder_name in folders
        if occursin(simulation_folder, simu_folder_name)
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
            nums = [4, 6]
            
            z_min = 1e-2
            if detector == "LIGO"
                N = Int(2e2)
                z_max = 5
            else
                N = Int(1e3)
                z_max = 20
            end
            z_range = exp10.(range(log10(z_min), stop = log10(z_max), length = N))
            file_path = "data/FisherErrorsRedshift_"*simu_folder_name*"_"*detector*"_"*string(convention)*"_"*string(label)*".h5"

            save_dict = Dict()

            @showprogress for M_f in masses
                for num_par in nums
                    save_dir = mode_1*"+"*mode_2*"/"*string(M_f)*"/"*string(num_par)
                    save_dict[save_dir] = zeros(N, 9)
                    Base.Threads.@threads for z_rand in z_range
                        i = findall(x -> x == z_rand, z_range)[1]
                        sigma, delta_var = RayleighCriterion(M_f, mass_f, F_Re, F_Im, num_par, mode_1, mode_2, noise, amplitudes, phases, ratios, dphases, omega, z_rand, convention)
                        save_results =  [z_rand, delta_var["f"], maximum(sigma["f_tau"]), delta_var["tau"], maximum(sigma["tau"]), delta_var["f"], maximum(sigma["f_Q"]), delta_var["Q"], maximum(sigma["Q"])]
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