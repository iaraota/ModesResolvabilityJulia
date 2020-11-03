#= The functions depend on the following modules (add them in the main)
using LinearAlgebra
include("AllFunctions.jl")
include("Constants.jl")
using .FrequencyTransforms, .PhysConstants, .Quantities
=#
function FisherMatrixElements(strain_unit, noise_freq, amplitudes, phases, freq, tau, mode_1, mode_2, convention = "FH")
	value = noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], amplitudes[mode_2] /amplitudes[mode_1], phases[mode_2], freq[mode_2], tau[mode_2]
	var_tau = "f0", "tau0", "f1", "tau1", "R", "dphi", "A0", "phi0"
	var_Q = "f0", "Q0", "f1", "Q1", "R", "dphi", "A0", "phi0"
    elements = Dict("tau"=>Dict("real"=> Dict(), "imaginary"=> Dict()), "Q" => Dict("real"=> Dict(), "imaginary"=> Dict()))
    
    if convention == "FH" || convention == "EF"
        for part in ["real", "imaginary"]
            for v in var_tau
                elements["tau"][part][v] = Fourier_d2modes.(v, part, value[1], value[2], value[3], value[4], value[5], value[6], value[7], value[8], value[9], "tau", convention)
            end
            for v in var_Q
                elements["Q"][part][v] = Fourier_d2modes.(v, part, value[1], value[2], value[3], value[4], value[5], value[6], value[7], value[8], value[9], "Q", convention)
            end
        end
    else
        error("convention argument must be set to \"FH\" or \"EF\".")
    end
	return elements, var_tau, var_Q
end

function RayleighCriterion(M_f, mass_f, F_Re, F_Im, num_par, mode_1, mode_2, noise, amplitudes, phases, omega, redshift, convention = "FH")
    # Computes parameters errors using Fisher Matrix 
    ## mode_1 is the dominant QNM and mode_2 is the next ratios = A_mode2/A_mode1 and dphases = phases_mode1 - phases_mode2

    Fisher_tau, Fisher_Q, elements_tau, elements_Q = Dict(), Dict(), Dict(), Dict() # fisher matrix dics and elements of fisher matrix
    Corr_tau, Corr_Q = Dict(), Dict() #inverse of fisher matrix
    sig_dftau, sig_dtau, sig_dfQ, sig_dQ = Dict(), Dict(), Dict(), Dict() # error = diagonal elements of Corr matrix

    # Source parameters
    M_final = (1+redshift)*M_f
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)

    time_unit = (M_total)*PhysConstants.tSun
    strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)

    freq, tau = Dict(), Dict()

    # Remnant BH frequency and decay time
    for (key, value) in omega
        freq[key] = value[1]/2/pi/time_unit
        tau[key] =  time_unit/value[2]
    end

    # define freq/tau/Q differences
    delta_var = Dict()
    delta_var["f"] = abs(freq[mode_1] - freq[mode_2])
    delta_var["tau"] = abs(tau[mode_1] - tau[mode_2])
    delta_var["Q"] = abs(pi * freq[mode_1] * tau[mode_1] - pi * freq[mode_2] * tau[mode_2])

    var_ind = Dict()
    elements, var_ind["tau"], var_ind["Q"] = FisherMatrixElements(strain_unit, noise["freq"], amplitudes, phases, freq, tau, mode_1, mode_2, convention)
    Fisher = Dict("tau"=>zeros(num_par, num_par), "Q"=> zeros(num_par, num_par))
    
    for i in 1:num_par
        for j in 1:num_par
        	for k in keys(Fisher)
            Fisher[k][i, j] = inner_product(noise["freq"], F_Re.*elements[k]["real"][var_ind[k][i]] + F_Im.*elements[k]["imaginary"][var_ind[k][i]], 
            	F_Re.*elements[k]["real"][var_ind[k][j]] + F_Im.*elements[k]["imaginary"][var_ind[k][j]], noise["psd"] .^ 2)
        	end
        end
    end

	Correlation = Dict()
    
    for k in keys(Fisher)
        if det(Fisher[k]) ≈ 0
            Correlation[k] = fill(1e30, (num_par, num_par))
        else   
        Correlation[k] = inv(Fisher[k])
        end
    end

	sigma = Dict()
	sigma["f_tau"] = [sqrt(abs(Correlation["tau"][1, 1])), sqrt(abs(Correlation["tau"][3, 3]))]
    sigma["tau"] = [sqrt(abs(Correlation["tau"][2, 2])), sqrt(abs(Correlation["tau"][4, 4]))]
    sigma["f_Q"] = [sqrt(abs(Correlation["Q"][1, 1])), sqrt(abs(Correlation["Q"][3, 3]))]
    sigma["Q"] = [sqrt(abs(Correlation["Q"][2, 2])), sqrt(abs(Correlation["Q"][4, 4]))]

    return sigma, delta_var
end

