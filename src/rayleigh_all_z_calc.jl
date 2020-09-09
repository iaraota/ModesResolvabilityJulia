using Distributions
using Random

include("fourier_derivs.jl") # import fourier transform of partial derivatives of the waveform
function elements_build(F_Re, F_Im, strain_unit, noise_freq, amplitudes, phases, ratios, dphases, freq, tau, mode_1, mode_2)
    # Return 8 parameters Fisher matrix elements [tau, Q]
    # F_Re and F_Im are the detector resaponse for each polarization
    #element parameters (f_220, tau_220/Q_220, f_k, tau_k/Q_k, R, dphi, A_0, phi_0)
    elements_tau_re = [
    ft_dh_df_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1]),
    ft_dh_dtau_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1]),
    ft_dh_df_Re.(noise_freq, ratios[mode_2]*amplitudes[mode_1] * strain_unit, phases[mode_1] - dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dtau_Re.(noise_freq, ratios[mode_2]*amplitudes[mode_1] * strain_unit, phases[mode_1] -dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dR_tau_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_ddphi1_tau_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2],dphases[mode_2],freq[mode_2], tau[mode_2]),
    ft_dh_dA0_tau_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dphitau_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], pi * freq[mode_1] * tau[mode_1])
    ]
    elements_Q_re = [
    ft_dh_dfQ_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], pi * freq[mode_1] * tau[mode_1]),
    ft_dh_dQ_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], pi * freq[mode_1] * tau[mode_1]),
    ft_dh_dfQ_Re.(noise_freq, ratios[mode_2]*amplitudes[mode_1] * strain_unit, phases[mode_1] - dphases[mode_2], freq[mode_2], pi * freq[mode_2] * tau[mode_2]),
    ft_dh_dQ_Re.(noise_freq, ratios[mode_2]*amplitudes[mode_1] * strain_unit, phases[mode_1] - dphases[mode_2],freq[mode_2], pi * freq[mode_2] * tau[mode_2]),
    ft_dh_dR_tau_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_ddphi1_tau_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dA0_tau_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dphiQ_Re.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], pi * freq[mode_1] * tau[mode_1])
    ]
    elements_tau_im = [
    ft_dh_df_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1]),
    ft_dh_dtau_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1]),
    ft_dh_df_Im.(noise_freq, ratios[mode_2]*amplitudes[mode_1] * strain_unit, phases[mode_1] - dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dtau_Im.(noise_freq,ratios[mode_2]*amplitudes[mode_1]* strain_unit, phases[mode_1] - dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dR_tau_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_ddphi1_tau_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2],dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dA0_tau_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dphitau_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], pi * freq[mode_1] * tau[mode_1])
    ]
    elements_Q_im = [
    ft_dh_dfQ_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], pi * freq[mode_1] * tau[mode_1]),
    ft_dh_dQ_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], pi * freq[mode_1] * tau[mode_1]),
    ft_dh_dfQ_Im.(noise_freq, ratios[mode_2]*amplitudes[mode_1] * strain_unit, phases[mode_1] - dphases[mode_2], freq[mode_2], pi * freq[mode_2] * tau[mode_2]),
    ft_dh_dQ_Im.(noise_freq, ratios[mode_2]*amplitudes[mode_1] * strain_unit, phases[mode_1] - dphases[mode_2],freq[mode_2], pi * freq[mode_2] * tau[mode_2]),
    ft_dh_dR_tau_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_ddphi1_tau_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2], dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dA0_tau_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], tau[mode_1], ratios[mode_2],dphases[mode_2], freq[mode_2], tau[mode_2]),
    ft_dh_dphiQ_Im.(noise_freq, amplitudes[mode_1] * strain_unit, phases[mode_1], freq[mode_1], pi * freq[mode_1] * tau[mode_1])
    ]
    return [F_Re*elements_tau_re + F_Im*elements_tau_im, F_Re*elements_Q_re + F_Im*elements_Q_im]
end


# mode_1 is the dominant QNM and mode_2 is the next ratios = A_mode2/A_mode1 and dphases = phases_mode1 - phases_mode2
function rayleight_calc(M_f, mass_f, F_Re, F_Im, num_par, mode_1, mode_2, noise, amplitudes, phases, ratios, dphases, omega, file_name)
    open(string(mode_2)*"_"*string(M_f)*file_name*".dat", "w") do save_file
        write(save_file, "# (1) redshift (2) df - max_sigma_f_tau (3) dtau - max_sigma_tau (4) df - max_sigma_f_Q (5) dQ - max_sigma_Q \n")
    end
    # constants
    Gconst = 6.67259e-11  # m^3/(kg*s^2)
    clight = 2.99792458e8  # m/s
    MSol = 1.989e30  # kg
    parsec = 3.08568025e16  # m

    # in seconds
    tSol = MSol*Gconst / clight^3  # from Solar mass to seconds
    Dist = 1.0e6*parsec / clight  # from Mpc to seconds

    resolv_keys = ["f_tau", "tau", "f_Q", "Q", "all"]
    z_resolv, z_f, z_i = Dict(), Dict(), Dict() #total z_resolv, min and max number to choose in the random step

    z_min, z_max = 1e-9, 20. # 0.04

    for key in resolv_keys
        z_resolv[key] = 0.0
        z_i[key] = z_min
        z_f[key] = z_max
    end
    delete!(z_resolv, "all")

    z_rand =  Dict() # the z_resolv of the wave for each loop, it will change until the freqs and taus are resolvable
    N = 1e3
    open(string(mode_2)*"_"*string(M_f)*"_"*string(num_par)*"_"*file_name*".dat", "a") do save_file
        for i in 1:N
            Fisher_tau, Fisher_Q, elements_tau, elements_Q = Dict(), Dict(), Dict(), Dict() # fisher matrix dics and elements of fisher matrix
            Corr_tau, Corr_Q = Dict(), Dict() #inverse of fisher matrix
            sig_dftau, sig_dtau, sig_dfQ, sig_dQ = Dict(), Dict(), Dict(), Dict() # error = diagonal elements of Corr matrix

            z_rand = (rand(Uniform(z_min, z_max)))
            # Source parameters
            M_final = (1+z_rand)*M_f
            M_total = M_final / mass_f
            D_L = luminosity_distance(z_rand)

            time_unit = (M_total)*tSol
            strain_unit = ((M_final)*tSol) / (D_L*Dist)

            freq, tau = Dict(), Dict()

            # Remnant BH frequency and decay time
            for (key, value) in omega
                freq[key] = value[1]/2/pi/time_unit
                tau[key] =  time_unit/value[2]
            end

            # define freq/tau/q difference
            delta_f, delta_tau, delta_Q = Dict(), Dict(), Dict()
            delta_f[mode_1*" - "*mode_2] = abs(freq[mode_1] - freq[mode_2])
            delta_tau[mode_1*" - "*mode_2] = abs(tau[mode_1] - tau[mode_2])
            delta_Q[mode_1*" - "*mode_2] = abs(pi * freq[mode_1] * tau[mode_1] - pi * freq[mode_2] * tau[mode_2])

            # define dictionaries for each mode k
            Fisher_tau, Fisher_Q, elements_tau, elements_Q = Dict(), Dict(), Dict(), Dict()
            Corr_tau, Corr_Q, Fisher_tau_high, Fisher_Q_high = Dict(), Dict(), Dict(), Dict()
            sig_dftau, sig_dtau, sig_dfQ, sig_dQ = Dict(), Dict(), Dict(), Dict()

            elements_tau, elements_Q = elements_build(F_Re, F_Im, strain_unit, noise["freq"], amplitudes, phases, ratios, dphases, freq, tau, mode_1, mode_2)
            Fisher_tau = zeros(num_par, num_par)
            Fisher_Q = zeros(num_par, num_par)


            for i in 1:num_par
                for j in 1:num_par
                    Fisher_tau[i, j] = inner_product(noise["freq"], elements_tau[i], elements_tau[j], noise["psd"] .^ 2)
                    Fisher_Q[i, j] = inner_product(noise["freq"], elements_Q[i], elements_Q[j], noise["psd"] .^ 2)
                end
            end

            Corr_tau = inv(Fisher_tau)
            Corr_Q = inv(Fisher_Q)

            sig_dftau = [sqrt(abs(Corr_tau[1, 1])), sqrt(abs(Corr_tau[3, 3]))]
            sig_dtau = [sqrt(abs(Corr_tau[2, 2])), sqrt(abs(Corr_tau[4, 4]))]
            sig_dfQ = [sqrt(abs(Corr_Q[1, 1])), sqrt(abs(Corr_Q[3, 3]))]
            sig_dQ = [sqrt(abs(Corr_Q[2, 2])), sqrt(abs(Corr_Q[4, 4]))]

            writedlm(save_file, [z_rand delta_f[mode_1*" - "*mode_2] - maximum(sig_dftau) delta_tau[mode_1*" - "*mode_2] - maximum(sig_dtau) delta_f[mode_1*" - "*mode_2] - maximum(sig_dfQ) delta_Q[mode_1*" - "*mode_2] - maximum(sig_dQ)])

        end
    end
end
