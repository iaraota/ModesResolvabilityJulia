__precompile__()
module FrequencyTransforms

    function QNM_reflec(t, A, ϕ, f, τ)
        return A*exp(-abs(t)/τ)*exp(1im*(2π*f*t + ϕ))
    end

    using FFTW, DSP

    function FFTReflectWindow(A, ϕ, f, τ, τ_220, strain_unit, time_unit, α = 1)
        # Compute discrete fourier transform of a QNM
        # Uses FFT, reflects the QNM and uses Tukey window
        # choose time interval for the QNM signal
        ## maximun time = 5*τ_220, amplitude is less than 0.1% the initial amplitude
        α = 1
        N = 2^14 - 1 
        t_max = 5*τ_220
        dt = 2*t_max/N
        t_all = -t_max:dt:t_max
        # compute QNM numerical signal
        ## real part
        signal_re = strain_unit*real.(QNM_reflec.(t_all, A, ϕ, f, τ)/2)
        signal_re = strain_unit*A*exp.(-abs.(t_all)/τ).*cos.(2π*f*t_all .+ ϕ)/2
        ## imaginary part
        signal_im = strain_unit*imag.(QNM_reflec.(t_all, A, ϕ, f, τ)/2)

        # Compute FFT
        ## FFT frequencies
        fft_freqs = FFTW.fftfreq(length(signal_re), 1.0/dt) |> fftshift
        ## tukey window
        tukey_window = DSP.Windows.tukey(length(signal_re), α)
        ## FFT of real part
        fft_re = dt*fft(tukey_window.*(signal_re)) |> fftshift
        ## FFT of imaginary part
        fft_im = dt*fft(tukey_window.*(signal_im)) |> fftshift

        return fft_re, fft_im, fft_freqs
    end

    function Fourier_1mode(part, f, A, φ, ωr, ωi, convention = "FH")
        ## compute the fourier transform of a single QNM
        ## fourier_QNM = \int_0^inf QNM
        ## QNM real part = A*exp(-ω_t*t)*cos(ω_r*t - φ)
        ## QNM imsaginary part = A*exp(-ω_t*t)*sin(ω_r*t - φ)
        ## part = "psd" returns the power spectrum of Re(QNM) + i Im(QNM)
        if convention == "EF"
            if part == "real"
                return A*((1im*2*pi*f + ωi)*cos(φ) + ωr*sin(φ)) / (ωr^2 - (2*pi*f - 1im*ωi)^2)
            elseif part == "imaginary"
                return -A*(-(1im*2*pi*f + ωi)*sin(φ) + ωr*cos(φ)) / (-ωr^2 + (2*pi*f - 1im*ωi)^2)
            elseif part == "psd"
                return A^2 / (ωi^2 + (ωr - 2*pi*f)^2) 
            else 
                error("fisrt argument of fourier_single_mode must be set to \"real\", \"imaginary\" or \"psd\".")
            end
        elseif convention == "FH"
            if part == "real"
                return A*ωi*(exp(1im*φ)/(ωi^2 + (-2π*f + ωr)^2) + exp(-1im*φ)/(ωi^2 + (2π*f + ωr)^2))/2
            elseif part == "imaginary"
                return 1im*A*ωi*(exp(1im*φ)/(ωi^2 + (-2π*f + ωr)^2) - exp(-1im*φ)/(ωi^2 + (2π*f + ωr)^2))/2
            else 
                error("fisrt argument of fourier_single_mode must be set to \"real\" or \"imaginary\".")
            end
        else
            error("convention argument must be set to \"FH\" or \"EF\".")
        end
    end 

    function Fourier_2QNMs(freq, amplitudes, phases, omega, mode_1, mode_2, time_unit, strain_unit, convention = "FH")
        ## Fourier transfor of two QNMs
        ## These are given in SI units
        ## time_unit and strain_unit, trasforms numerial units in SI units
        ## returns dictionaries for real and imaginary fourier transforms of QNM1, QNM2 and QNM1 + QNM2
        ft_Re, ft_Im = Dict(), Dict()

        if convention == "FH" || convention == "EF"
            ft_Re[mode_1] = time_unit*strain_unit*abs.(Fourier_1mode.("real", freq.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], convention))
            ft_Re[mode_2] = time_unit*strain_unit*abs.(Fourier_1mode.("real", freq.*time_unit, amplitudes[mode_2], phases[mode_2], omega[mode_2][1], omega[mode_2][2], convention))
            ft_Re[mode_1*" + "*mode_2] = abs.(Fourier_1mode.("real", freq.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], convention) +
            Fourier_1mode.("real", freq*time_unit, amplitudes[mode_2], phases[mode_2], omega[mode_2][1], omega[mode_2][2], convention))*time_unit*strain_unit

            ft_Im[mode_1] = time_unit*strain_unit*abs.(Fourier_1mode.("imaginary", freq.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], convention))
            ft_Im[mode_2] = time_unit*strain_unit*abs.(Fourier_1mode.("imaginary", freq.*time_unit, amplitudes[mode_2], phases[mode_2], omega[mode_2][1], omega[mode_2][2], convention))
            ft_Im[mode_1*" + "*mode_2] = abs.(Fourier_1mode.("imaginary", freq.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], convention) +
            Fourier_1mode.("imaginary", freq.*time_unit, amplitudes[mode_2], phases[mode_2], omega[mode_2][1], omega[mode_2][2], convention))*time_unit*strain_unit    
            
            return ft_Re, ft_Im
        elseif convention == "DFT"
            ft_freqs = Dict()
            for mode in [mode_1, mode_2]
                ft_Re[mode], ft_Im[mode], ft_freqs[mode] = FFTReflectWindow(amplitudes[mode], phases[mode], omega[mode][1]/2π/time_unit, time_unit/omega[mode][2], time_unit/omega["(2,2,0)"][2], strain_unit, time_unit, 1)
            end
            ft_Re[mode_1*" + "*mode_2] = abs.(ft_Re[mode_1] .+ ft_Re[mode_2])
            ft_Im[mode_1*" + "*mode_2] = abs.(ft_Im[mode_1] .+ ft_Im[mode_2])
            ft_freqs[mode_1*" + "*mode_2] = ft_freqs[mode_1]

            for mode in [mode_1, mode_2]
                ft_Re[mode], ft_Im[mode] = abs.(ft_Re[mode]), abs.(ft_Im[mode])
            end
            return ft_Re, ft_Im, ft_freqs
        else
            error("convention argument must be set to \"FH\", \"EF\" or \"DFT\".")
        end
        
    end

    # fourier transform of the partial derivative of the QNM 
    # used to compute Fisher matrix 
    # eqs. (4.4) and (7.4) of https://arxiv.org/pdf/gr-qc/0512160.pdf
    include("FourierTransformsPartial.jl")
    using .FourierTransformsPartial, .FourierTransformsPartialFH
    function Fourier_d1mode(variable, part, freq, A, φ, f, τ, decay = "tau", convention = "FH")
        ## Consider modes are written independently: 
        ### Re(QNM) = A0*exp(-ωi0*t)*cos(ωr0*t - φ0) + A1*exp(-ωi1*t)*cos(ωr1*t - φ1)
        ### Im(QNM) = A0*exp(-ωi0*t)*sin(ωr0*t - φ0) + A1*exp(-ωi1*t)*sin(ωr1*t - φ1)
        if convention == "FH"
            if decay == "tau"
                if part == "real"
                    if variable == "f"
                        return FourierTransformsPartialFH.ft_dh_df_Re(freq, A, φ, f, τ)
                    elseif variable == "tau"
                        return FourierTransformsPartialFH.ft_dh_dtau_Re(freq, A, φ, f, τ)
                    elseif variable == "A"
                        return FourierTransformsPartialFH.ft_dh_dAtau_Re(freq, A, φ, f, τ)
                    elseif variable == "phi"
                        return FourierTransformsPartialFH.ft_dh_dphitau_Re(freq, A, φ, f, τ)
                    else 
                        error("Fourier_d1mode first argument should be \"f\", \"tau\", \"A\", \"phi\".")
                    end
                elseif part == "imaginary"
                    if variable == "f"
                        return FourierTransformsPartialFH.ft_dh_df_Im(freq, A, φ, f, τ)
                    elseif variable == "tau"
                        return FourierTransformsPartialFH.ft_dh_dtau_Im(freq, A, φ, f, τ)
                    elseif variable == "A"
                        return FourierTransformsPartialFH.ft_dh_dAtau_Im(freq, A, φ, f, τ)
                    elseif variable == "phi"
                        return FourierTransformsPartialFH.ft_dh_dphitau_Im(freq, A, φ, f, τ)
                    else 
                        error("Fourier_d1mode first argument should be \"f\", \"tau\", \"A\", \"phi\".")
                    end
                else 
                    error("Fourier_d1mode second argument should be \"real\" or \"imaginary\".")
                end
            elseif decay == "Q"
                # quality factor Q = π*f*τ
                if part == "real"
                    if variable == "f"
                        return FourierTransformsPartialFH.ft_dh_dfQ_Re(freq, A, φ, f, pi*f*τ)
                    elseif variable == "Q"
                        return FourierTransformsPartialFH.ft_dh_dQ_Re(freq, A, φ, f, pi*f*τ)
                    elseif variable == "A"
                        return FourierTransformsPartialFH.ft_dh_dAQ_Re(freq, A, φ, f, pi*f*τ)
                    elseif variable == "phi"
                        return FourierTransformsPartialFH.ft_dh_dphiQ_Re(freq, A, φ, f, pi*f*τ)
                    else 
                        error("Fourier_d1mode first argument should be \"f\", \"Q\", \"A\", \"phi\".")
                    end
                elseif part == "imaginary"
                    if variable == "f"
                        return FourierTransformsPartialFH.ft_dh_dfQ_Im(freq, A, φ, f, pi*f*τ)
                    elseif variable == "Q"
                        return FourierTransformsPartialFH.ft_dh_dQ_Im(freq, A, φ, f, pi*f*τ)
                    elseif variable == "A"
                        return FourierTransformsPartialFH.ft_dh_dAQ_Im(freq, A, φ, f, pi*f*τ)
                    elseif variable == "phi"
                        return FourierTransformsPartialFH.ft_dh_dphiQ_Im(freq, A, φ, f, pi*f*τ)
                    else 
                        error("Fourier_d1mode first argument should be \"f\", \"Q\", \"A\", \"phi\".")
                    end
                else 
                    error("Fourier_d1mode second argument should be \"real\" or \"imaginary\".")
                end
            else 
                error("Fourier_d1mode last argument should be \"tau\" or \"Q\".")
            end
        elseif convention == "EF"
            if decay == "tau"
                if part == "real"
                    if variable == "f"
                        return FourierTransformsPartial.ft_dh_df_Re(freq, A, φ, f, τ)
                    elseif variable == "tau"
                        return FourierTransformsPartial.ft_dh_dtau_Re(freq, A, φ, f, τ)
                    elseif variable == "A"
                        return FourierTransformsPartial.ft_dh_dAtau_Re(freq, A, φ, f, τ)
                    elseif variable == "phi"
                        return FourierTransformsPartial.ft_dh_dphitau_Re(freq, A, φ, f, τ)
                    else 
                        error("Fourier_d1mode first argument should be \"f\", \"tau\", \"A\", \"phi\".")
                    end
                elseif part == "imaginary"
                    if variable == "f"
                        return FourierTransformsPartial.ft_dh_df_Im(freq, A, φ, f, τ)
                    elseif variable == "tau"
                        return FourierTransformsPartial.ft_dh_dtau_Im(freq, A, φ, f, τ)
                    elseif variable == "A"
                        return FourierTransformsPartial.ft_dh_dAtau_Im(freq, A, φ, f, τ)
                    elseif variable == "phi"
                        return FourierTransformsPartial.ft_dh_dphitau_Im(freq, A, φ, f, τ)
                    else 
                        error("Fourier_d1mode first argument should be \"f\", \"tau\", \"A\", \"phi\".")
                    end
                else 
                    error("Fourier_d1mode second argument should be \"real\" or \"imaginary\".")
                end
            elseif decay == "Q"
                # quality factor Q = π*f*τ
                if part == "real"
                    if variable == "f"
                        return FourierTransformsPartial.ft_dh_dfQ_Re(freq, A, φ, f, pi*f*τ)
                    elseif variable == "Q"
                        return FourierTransformsPartial.ft_dh_dQ_Re(freq, A, φ, f, pi*f*τ)
                    elseif variable == "A"
                        return FourierTransformsPartial.ft_dh_dAQ_Re(freq, A, φ, f, pi*f*τ)
                    elseif variable == "phi"
                        return FourierTransformsPartial.ft_dh_dphiQ_Re(freq, A, φ, f, pi*f*τ)
                    else 
                        error("Fourier_d1mode first argument should be \"f\", \"Q\", \"A\", \"phi\".")
                    end
                elseif part == "imaginary"
                    if variable == "f"
                        return FourierTransformsPartial.ft_dh_dfQ_Im(freq, A, φ, f, pi*f*τ)
                    elseif variable == "Q"
                        return FourierTransformsPartial.ft_dh_dQ_Im(freq, A, φ, f, pi*f*τ)
                    elseif variable == "A"
                        return FourierTransformsPartial.ft_dh_dAQ_Im(freq, A, φ, f, pi*f*τ)
                    elseif variable == "phi"
                        return FourierTransformsPartial.ft_dh_dphiQ_Im(freq, A, φ, f, pi*f*τ)
                    else 
                        error("Fourier_d1mode first argument should be \"f\", \"Q\", \"A\", \"phi\".")
                    end
                else 
                    error("Fourier_d1mode second argument should be \"real\" or \"imaginary\".")
                end
            else 
                error("Fourier_d1mode last argument should be \"tau\" or \"Q\".")
            end
        else
            error("convention argument must be set to \"FH\" or \"EF\".")
        end
    end

    function Fourier_d2modes(variable, part, freq, A0, φ0, f0, τ0, R, φ1, f1, τ1, decay = "tau", convention = "FH")
        ## Consider modes overall amplitude A0
        ## Re(QNM) = A0*(exp(-ωi0*t)*cos(ωr0*t - φ0) + R*exp(-ωi1*t)*cos(ωr1*t - φ1))
        if convention == "FH"
            if decay == "tau"
                if part == "real"
                    if variable == "f0"
                        return FourierTransformsPartialFH.ft_dh_df_Re(freq, A0, φ0, f0, τ0)
                    elseif variable == "tau0"
                        return FourierTransformsPartialFH.ft_dh_dtau_Re(freq, A0, φ0, f0, τ0)
                    elseif variable == "f1"
                        return FourierTransformsPartialFH.ft_dh_df_Re(freq, R/A0, φ1, f1, τ1)
                    elseif variable == "tau1"
                        return FourierTransformsPartialFH.ft_dh_dtau_Re(freq, R/A0, φ1, f1, τ1)
                    elseif variable == "R"
                        return FourierTransformsPartialFH.ft_dh_dR_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "dphi"
                        return FourierTransformsPartialFH.ft_dh_ddphi1_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "A0"
                        return FourierTransformsPartialFH.ft_dh_dA0_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "phi0"
                        return FourierTransformsPartialFH.ft_dh_dphitau_Re(freq, A0, φ0, f0, τ0)
                    else 
                        error("Fourir_d2modes first argument should be: \n 
                            \"f0\", \"tau0\", \"f1\", \"tau1\", \"R\", \"dphi\", \"A0\", \"phi0\".")
                    end
                elseif part == "imaginary"
                    if variable == "f0"
                        return FourierTransformsPartialFH.ft_dh_df_Im(freq, A0, φ0, f0, τ0)
                    elseif variable == "tau0"
                        return FourierTransformsPartialFH.ft_dh_dtau_Im(freq, A0, φ0, f0, τ0)
                    elseif variable == "f1"
                        return FourierTransformsPartialFH.ft_dh_df_Im(freq, R/A0, φ1, f1, τ1)
                    elseif variable == "tau1"
                        return FourierTransformsPartialFH.ft_dh_dtau_Im(freq, R/A0, φ1, f1, τ1)
                    elseif variable == "R"
                        return FourierTransformsPartialFH.ft_dh_dR_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "dphi"
                        return FourierTransformsPartialFH.ft_dh_ddphi1_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "A0"
                        return FourierTransformsPartialFH.ft_dh_dA0_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "phi0"
                        return FourierTransformsPartialFH.ft_dh_dphitau_Im(freq, A0, φ0, f0, τ0)
                    else 
                        error("Fourir_d2modes first argument should be: \n 
                            \"f0\", \"tau0\", \"f1\", \"tau1\", \"R\", \"dphi\", \"A0\", \"phi0\".")
                    end
                else 
                    error("Fourir_d2modes second argument should be \"real\" or \"imaginary\".")
                end
            elseif decay == "Q"
                # quality factor Q = π*f*tau
                if part == "real"
                    if variable == "f0"
                        return FourierTransformsPartialFH.ft_dh_dfQ_Re(freq, A0, φ0, f0, pi*f0*τ0)
                    elseif variable == "Q0"
                        return FourierTransformsPartialFH.ft_dh_dQ_Re(freq, A0, φ0, f0, pi*f0*τ0)
                    elseif variable == "f1"
                        return FourierTransformsPartialFH.ft_dh_dfQ_Re(freq, R/A0, φ1, f1, pi*f1*τ1)
                    elseif variable == "Q1"
                        return FourierTransformsPartialFH.ft_dh_dQ_Re(freq, R/A0, φ1, f1, pi*f1*τ1)
                    elseif variable == "R"
                        return FourierTransformsPartialFH.ft_dh_dR_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "dphi"
                        return FourierTransformsPartialFH.ft_dh_ddphi1_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "A0"
                        return FourierTransformsPartialFH.ft_dh_dA0_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "phi0"
                        return FourierTransformsPartialFH.ft_dh_dphitau_Re(freq, A0, φ0, f0, τ0)
                    else 
                        error("Fourir_d2modes first argument should be: \n 
                            \"f0\", \"tau0\", \"f1\", \"tau1\", \"R\", \"dphi\", \"A0\", \"phi0\".")
                    end
                elseif part == "imaginary"
                    if variable == "f0"
                        return FourierTransformsPartialFH.ft_dh_dfQ_Im(freq, A0, φ0, f0, pi*f0*τ0)
                    elseif variable == "Q0"
                        return FourierTransformsPartialFH.ft_dh_dQ_Im(freq, A0, φ0, f0, pi*f0*τ0)
                    elseif variable == "f1"
                        return FourierTransformsPartialFH.ft_dh_dfQ_Im(freq, R/A0, φ1, f1, pi*f1*τ1)
                    elseif variable == "Q1"
                        return FourierTransformsPartialFH.ft_dh_dQ_Im(freq, R/A0, φ1, f1, pi*f1*τ1)
                    elseif variable == "R"
                        return FourierTransformsPartialFH.ft_dh_dR_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "dphi"
                        return FourierTransformsPartialFH.ft_dh_ddphi1_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "A0"
                        return FourierTransformsPartialFH.ft_dh_dA0_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "phi0"
                        return FourierTransformsPartialFH.ft_dh_dphitau_Im(freq, A0, φ0, f0, τ0)
                    else 
                        error("Fourir_d2modes first argument should be: \n 
                            \"f0\", \"tau0\", \"f1\", \"tau1\", \"R\", \"dphi\", \"A0\", \"phi0\".")
                    end
                else 
                    error("Fourir_d2modes second argument should be \"real\" or \"imaginary\".")
                end
            else 
                error("Fourir_d2modes last argument should be \"tau\" or \"Q\".")
            end
        elseif convention == "EF"
            if decay == "tau"
                if part == "real"
                    if variable == "f0"
                        return FourierTransformsPartial.ft_dh_df_Re(freq, A0, φ0, f0, τ0)
                    elseif variable == "tau0"
                        return FourierTransformsPartial.ft_dh_dtau_Re(freq, A0, φ0, f0, τ0)
                    elseif variable == "f1"
                        return FourierTransformsPartial.ft_dh_df_Re(freq, R/A0, φ1, f1, τ1)
                    elseif variable == "tau1"
                        return FourierTransformsPartial.ft_dh_dtau_Re(freq, R/A0, φ1, f1, τ1)
                    elseif variable == "R"
                        return FourierTransformsPartial.ft_dh_dR_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "dphi"
                        return FourierTransformsPartial.ft_dh_ddphi1_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "A0"
                        return FourierTransformsPartial.ft_dh_dA0_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "phi0"
                        return FourierTransformsPartial.ft_dh_dphitau_Re(freq, A0, φ0, f0, τ0)
                    else 
                        error("Fourir_d2modes first argument should be: \n 
                            \"f0\", \"tau0\", \"f1\", \"tau1\", \"R\", \"dphi\", \"A0\", \"phi0\".")
                    end
                elseif part == "imaginary"
                    if variable == "f0"
                        return FourierTransformsPartial.ft_dh_df_Im(freq, A0, φ0, f0, τ0)
                    elseif variable == "tau0"
                        return FourierTransformsPartial.ft_dh_dtau_Im(freq, A0, φ0, f0, τ0)
                    elseif variable == "f1"
                        return FourierTransformsPartial.ft_dh_df_Im(freq, R/A0, φ1, f1, τ1)
                    elseif variable == "tau1"
                        return FourierTransformsPartial.ft_dh_dtau_Im(freq, R/A0, φ1, f1, τ1)
                    elseif variable == "R"
                        return FourierTransformsPartial.ft_dh_dR_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "dphi"
                        return FourierTransformsPartial.ft_dh_ddphi1_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "A0"
                        return FourierTransformsPartial.ft_dh_dA0_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "phi0"
                        return FourierTransformsPartial.ft_dh_dphitau_Im(freq, A0, φ0, f0, τ0)
                    else 
                        error("Fourir_d2modes first argument should be: \n 
                            \"f0\", \"tau0\", \"f1\", \"tau1\", \"R\", \"dphi\", \"A0\", \"phi0\".")
                    end
                else 
                    error("Fourir_d2modes second argument should be \"real\" or \"imaginary\".")
                end
            elseif decay == "Q"
                # quality factor Q = π*f*tau
                if part == "real"
                    if variable == "f0"
                        return FourierTransformsPartial.ft_dh_dfQ_Re(freq, A0, φ0, f0, pi*f0*τ0)
                    elseif variable == "Q0"
                        return FourierTransformsPartial.ft_dh_dQ_Re(freq, A0, φ0, f0, pi*f0*τ0)
                    elseif variable == "f1"
                        return FourierTransformsPartial.ft_dh_dfQ_Re(freq, R/A0, φ1, f1, pi*f1*τ1)
                    elseif variable == "Q1"
                        return FourierTransformsPartial.ft_dh_dQ_Re(freq, R/A0, φ1, f1, pi*f1*τ1)
                    elseif variable == "R"
                        return FourierTransformsPartial.ft_dh_dR_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "dphi"
                        return FourierTransformsPartial.ft_dh_ddphi1_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "A0"
                        return FourierTransformsPartial.ft_dh_dA0_tau_Re(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "phi0"
                        return FourierTransformsPartial.ft_dh_dphitau_Re(freq, A0, φ0, f0, τ0)
                    else 
                        error("Fourir_d2modes first argument should be: \n 
                            \"f0\", \"tau0\", \"f1\", \"tau1\", \"R\", \"dphi\", \"A0\", \"phi0\".")
                    end
                elseif part == "imaginary"
                    if variable == "f0"
                        return FourierTransformsPartial.ft_dh_dfQ_Im(freq, A0, φ0, f0, pi*f0*τ0)
                    elseif variable == "Q0"
                        return FourierTransformsPartial.ft_dh_dQ_Im(freq, A0, φ0, f0, pi*f0*τ0)
                    elseif variable == "f1"
                        return FourierTransformsPartial.ft_dh_dfQ_Im(freq, R/A0, φ1, f1, pi*f1*τ1)
                    elseif variable == "Q1"
                        return FourierTransformsPartial.ft_dh_dQ_Im(freq, R/A0, φ1, f1, pi*f1*τ1)
                    elseif variable == "R"
                        return FourierTransformsPartial.ft_dh_dR_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "dphi"
                        return FourierTransformsPartial.ft_dh_ddphi1_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "A0"
                        return FourierTransformsPartial.ft_dh_dA0_tau_Im(freq, A0, φ0, f0, τ0, R, φ0 - φ1, f1, τ1)
                    elseif variable == "phi0"
                        return FourierTransformsPartial.ft_dh_dphitau_Im(freq, A0, φ0, f0, τ0)
                    else 
                        error("Fourir_d2modes first argument should be: \n 
                            \"f0\", \"tau0\", \"f1\", \"tau1\", \"R\", \"dphi\", \"A0\", \"phi0\".")
                    end
                else 
                    error("Fourir_d2modes second argument should be \"real\" or \"imaginary\".")
                end
            else 
                error("Fourir_d2modes last argument should be \"tau\" or \"Q\".")
            end
        else
            error("convention argument must be set to \"FH\" or \"EF\".")
        end
    end

    export Fourier_1mode, PowerSpectrum_2modes, Fourier_2QNMs, Fourier_d2modes
end