module FourierTransformsPartial
    # fourier transform of the derivative of the strain
    ## used to compute Fisher matrix 
    ## eqs. (4.4) and (7.4) of https://arxiv.org/pdf/gr-qc/0512160.pdf
    ##relative to frequency f (with tau)
    ### real part
    function ft_dh_df_Re(f, A, phi, f0, tau)
        return 1im * A * exp(-1im * phi) * pi * tau ^ 2 *
                (exp(1im * 2 * phi) / (1im - 2 * pi * (f + f0) * tau) ^ 2 -
                1 / (1im + 2 * (f0 - f) * pi * tau) ^ 2)
    end

    ### imaginary part
    function ft_dh_df_Im(f, A, phi, f0, tau)
        return -A * exp(-1im * phi) * pi * tau ^ 2 *
                (exp(1im * 2 * phi) / (1im - 2 * pi * (f + f0) * tau) ^ 2 +
                1 / (1im + 2 * (f0 - f) * pi * tau) ^ 2)
    end

    ## relative to decay time tau
    ### real part
    function ft_dh_dtau_Re(f, A, phi, f0, tau)
        return -(1 / 2) * A * exp(-1im * phi) * (1 / (1im + 2 * (f0 - f) * pi * tau) ^ 2 +
                                                    exp(1im * 2 * phi) / (1im - 2 * (f + f0) * pi * tau) ^ 2)
    end

    ### imaginary part
    function ft_dh_dtau_Im(f, A, phi, f0, tau)
        return (1im / 2) * A * exp(-1im * phi) * (1 / (1im + 2 * (f0 - f) * pi * tau) ^ 2 -
                                                    exp(1im * 2 * phi) / (1im - 2 * (f + f0) * pi * tau) ^ 2)
    end

    ## relative to amplitude A
    function ft_dh_dAtau_Re(f, A, phi, f0, tau)
        return (1im * exp(-1im * phi) * tau / (2im + 4 * (f0 - f) * pi * tau) - 1im * exp(1im * phi) * tau / (
                    -2im + 4 * (f0 + f) * pi * tau))
    end

    function ft_dh_dAtau_Im(f, A, phi, f0, tau)
        return (exp(-1im * phi) * tau / (2im + 4 * (f0 - f) * pi * tau) + exp(1im * phi) * tau / (
                    -2im + 4 * (f0 + f) * pi * tau))
    end

    function ft_dh_dA0_tau_Re(f, A0, phi0, f0, tau0, R, dphi1, f1, tau1)
        return -(1im / 2) * exp(-1im * phi0) * (tau0 / (-1im + 2 * (f - f0) * pi * tau0)
                                                    + exp(1im * dphi1) * R * tau1 / (-1im + 2 * (f - f1) * pi * tau1))
                -(1im / 2) * exp(1im * phi0) * (tau0 / (-1im + 2 * (f + f0) * pi * tau0)
                                                + exp(-1im * dphi1) * R * tau1 / (-1im + 2 * (f + f1) * pi * tau1))
    end

    function ft_dh_dA0_tau_Im(f, A0, phi0, f0, tau0, R, dphi1, f1, tau1)
        return exp(1im * phi0) * (tau0 / (2im + 4 * (-f + f0) * pi * tau0)
                                    + exp(1im * dphi1) * R * tau1 / (2im + 4 * (-f + f1) * pi * tau1))
                +(1 / 2) * exp(1im * phi0) * (tau0 / (-1im + 2 * (f + f0) * pi * tau0)
                                                + exp(-1im * dphi1) * R * tau1 / (-1im + 2 * (f + f1) * pi * tau1))
    end


    function ft_dh_dR_tau_Re(f, A0, phi0, f0, tau0, R, dphi1, f1, tau1)
        return (1im * A0 * exp(1im * (dphi1 - phi0)) * tau1) / (2im + 4 * (-f + f1) * pi * tau1)
                - 1im * A0 * exp(1im * (phi0 - dphi1)) * tau1 / (-2im + 4 * (f + f1) * pi * tau1)
    end

    function ft_dh_dR_tau_Im(f, A0, phi0, f0, tau0, R, dphi1, f1, tau1)
        return A0 * exp(1im * (dphi1 - phi0)) * tau1 / (2im + 4 * (-f + f1) * pi * tau1)
                + A0 * exp(1im * (phi0 - dphi1)) * tau1 / (2 * (-1im + 2 * (f + f1) * pi * tau1))
    end

    function ft_dh_ddphi1_tau_Re(f, A0, phi0, f0, tau0, R, dphi1, f1, tau1)
        return A0 * exp(1im * (dphi1 - phi0)) * R * tau1 / (-2im + 4 * (f - f1) * pi * tau1)
                + A0 * exp(1im * (phi0 - dphi1)) * R * tau1 / (2im - 4 * (f + f1) * pi * tau1)
    end

    function ft_dh_ddphi1_tau_Im(f, A0, phi0, f0, tau0, R, dphi1, f1, tau1)
        return 1im * A0 * exp(1im * dphi1 - 1im * phi0) * R * tau1 / (2im + 4 * (-f + f1) * pi * tau1)
                -1im * A0 * exp(-1im * dphi1 + 1im * phi0) * R * tau1 / (2 * (-1im + 2 * (f + f1) * pi * tau1))
    end

    ## relative to phase phi
    function ft_dh_dphitau_Re(f, A, phi, f0, tau)
        return A * exp(-1im * phi) * tau / (2im + 4 * (f0 - f) * pi * tau) + A * exp(1im * phi) * tau / (
                    -2im + 4 * (f + f0) * pi * tau)
    end

    function ft_dh_dphitau_Im(f, A, phi, f0, tau)
        return 1im * A * exp(-1im * phi) * tau / (-2im + 4 * (f - f0) * pi * tau) + 1im * A * exp(1im * phi) * tau / (
                    -2im + 4 * (f + f0) * pi * tau)
    end

    ## relative to the frequency f (with quality factor)
    ### real part
    function ft_dh_dfQ_Re(f, A, phi, f0, Q)
        return A * exp(1im * phi) * (1 + 1im * 2 * Q) * Q / (
                2 * pi * (1im * f0 - 2 * (f + f0) * Q) ^ 2) + A * exp(-1im * phi) * (1 - 2 * 1im * Q) * Q / (
                        2 * pi * (-2 * f * Q + f0 * (1im + 2 * Q)) ^ 2)
    end

    function ft_dh_dfQ_Im(f, A, phi, f0, Q)
        return A * exp(1im * phi) * (1im - 2 * Q) * Q / (2 * pi * (1im * f0 - 2 * (f + f0) * Q) ^ 2) -
                A * exp(-1im * phi) * Q * (1im + 2 * Q) / (2 * pi * (-2 * f * Q + f0 * (1im + 2 * Q)) ^ 2)
    end

    ## relative to quality factor Q = pi*f*tau
    ### real part
    function ft_dh_dQ_Re(f, A, phi, f0, Q)
        return -A * exp(1im * phi) * f0 / (2 * pi * (1im * f0 - 2 * (f + f0) * Q) ^ 2) - A * exp(
            -1im * phi) * f0 / (2 * pi * (-2 * f * Q + f0 * (1im + 2 * Q)) ^ 2)
    end

    function ft_dh_dQ_Im(f, A, phi, f0, Q)
        return -1im * A * exp(1im * phi) * f0 / (2 * pi * (1im * f0 - 2 * (f + f0) * Q) ^ 2) +
                1im * A * exp(-1im * phi) * f0 / (2 * pi * (-2 * f * Q + f0 * (1im + 2 * Q)) ^ 2)
    end

    ## relative to amplitude A
    function ft_dh_dAQ_Re(f, A, phi, f0, Q)
        return (1im * exp(-1im * phi) * Q / (2im * f0 * pi + 4 * (f0 - f) * pi * Q) - 1im * exp(1im * phi) * Q / (
                    -2im * f0 * pi + 4 * (f0 + f) * pi * Q))
    end

    function ft_dh_dAQ_Im(f, A, phi, f0, Q)
        return (exp(-1im * phi) * Q / (2im * f0 * pi + 4 * (f0 - f) * pi * Q) + exp(1im * phi) * Q / (
                    -2im * f0 * pi + 4 * (f0 + f) * pi * Q))
    end

    ## relative to phase phi
    function ft_dh_dphiQ_Re(f, A, phi, f0, Q)
        return A * exp(-1im * phi) * Q / (2im * f0 * pi + 4 * (f0 - f) * pi * Q) + A * exp(1im * phi) * Q / (
                    -2im * f0 * pi + 4 * (f0 + f) * pi * Q)
    end

    function ft_dh_dphiQ_Im(f, A, phi, f0, Q)
        return -1im * A * exp(-1im * phi) * Q / (2im * f0 * pi + 4 * (f0 - f) * pi * Q) + 1im * A * exp(
            1im * phi) * Q / (-2im * f0 * pi + 4 * (f0 + f) * pi * Q)
    end
end

module FourierTransformsPartialFH
    # fourier transform of the derivative of the strain
    ## used to compute Fisher matrix 
    ## eqs. (4.4) and (7.4) of https://arxiv.org/pdf/gr-qc/0512160.pdf
    ##relative to frequency f (with tau)
    ### real part
    function ft_dh_df_Re(f, A_lmn, ϕ_lmn, f_lmn, τ_lmn)
        return 8*A_lmn*exp(-1im*ϕ_lmn)*π^2*τ_lmn^3*(
            exp(2im*ϕ_lmn)*(f - f_lmn)/(1 + (2π*τ_lmn*(f - f_lmn))^2)^2
            - (f + f_lmn)/(1 + (2π*τ_lmn*(f + f_lmn))^2)^2
            )/2
    end

    ### imaginary part
    function ft_dh_df_Im(f, A_lmn, ϕ_lmn, f_lmn, τ_lmn)
        return 8im*A_lmn*exp(-1im*ϕ_lmn)*π^2*τ_lmn^3*(
            exp(2im*ϕ_lmn)*(f - f_lmn)/(1 + (2π*τ_lmn*(f - f_lmn))^2)^2 
            + (f + f_lmn)/(1 + (2π*τ_lmn*(f + f_lmn))^2)^2 
        )/2
    end

    ## relative to decay time tau
    ### real part
    function ft_dh_dtau_Re(f, A_lmn, ϕ_lmn, f_lmn, τ_lmn)
        return A_lmn*exp(-1im*ϕ_lmn)*(exp(2im*ϕ_lmn)*(
            1 - (2π*τ_lmn*(f - f_lmn))^2)/(1 + (2π*τ_lmn*(f - f_lmn))^2)^2 
            + (1 - (2π*τ_lmn*(f + f_lmn))^2)/(1 + (2π*τ_lmn*(f + f_lmn))^2)^2
            )/2
    end

    ### imaginary part
    function ft_dh_dtau_Im(f, A_lmn, ϕ_lmn, f_lmn, τ_lmn)
        return -1im*A_lmn*exp(-1im*ϕ_lmn)*(
            exp(2im*ϕ_lmn)*(-1 + 2π*τ_lmn*(f - f_lmn))*(1 + 2π*τ_lmn*(f - f_lmn))/(1 + (2π*τ_lmn*(f - f_lmn))^2)^2
            + (1 - (2π*τ_lmn*(f + f_lmn))^2)/ (1 + (2π*τ_lmn*(f + f_lmn))^2)^2
        )/2
    end

    ## relative to amplitude A
    function ft_dh_dAtau_Re(f, A_lmn, ϕ_lmn, f_lmn, τ_lmn)
        return τ_lmn*(exp(1im*ϕ_lmn)/(1 + (2π*τ_lmn*(f - f_lmn))^2) + exp(-1im*ϕ_lmn)/(1 + (2π*τ_lmn*(f + f_lmn))^2))/2
    end

    function ft_dh_dAtau_Im(f, A_lmn, ϕ_lmn, f_lmn, τ_lmn)
        return 1im*τ_lmn*(exp(1im*ϕ_lmn)/(1 + (2π*τ_lmn*(f - f_lmn))^2) - exp(-1im*ϕ_lmn)/(1 + (2π*τ_lmn*(f + f_lmn))^2))/2
    end

    ## relative to phase phi
    function ft_dh_dphitau_Re(f, A_lmn, ϕ_lmn, f_lmn, τ_lmn)
        return 1im*A_lmn*τ_lmn*(exp(1im*ϕ_lmn)/(1 + (2π*τ_lmn*(f - f_lmn))^2) - exp(-1im*ϕ_lmn)/(1 + (2π*τ_lmn*(f + f_lmn))^2))/2
    end

    function ft_dh_dphitau_Im(f, A_lmn, ϕ_lmn, f_lmn, τ_lmn)
        return -A_lmn*τ_lmn*(exp(1im*ϕ_lmn)/(1 + (2π*τ_lmn*(f - f_lmn))^2) + exp(-1im*ϕ_lmn)/(1 + (2π*τ_lmn*(f + f_lmn))^2))/2
    end

    ## Two modes as function of amplitude ratio R and phase difference dphi
    ## relative to amplitude A 
    function ft_dh_dA0_tau_Re(f, A_0, ϕ_0, f_0, τ_0, R, dϕ, f_1, τ_1)
        return exp(-1im*ϕ_0)*(
            exp(2im*ϕ_0)*τ_0/(1 + (2π*τ_0*(f - f_0))^2) +
            τ_0/(1 + (2π*τ_0*(f + f_0))^2) + 
            exp(-1im*(dϕ - 2*ϕ_0))*R*τ_1/(1 + (2π*τ_1*(f - f_1))^2) + 
            exp(1im*dϕ)*R*τ_1/(1 + (2π*τ_1*(f + f_1))^2)
        )/2
    end

    function ft_dh_dA0_tau_Im(f, A_0, ϕ_0, f_0, τ_0, R, dϕ, f_1, τ_1)
        return 1im*exp(-1im*ϕ_0)*(
            exp(2im*ϕ_0)*τ_0/(1 + (2π*τ_0*(f - f_0))^2) -
            τ_0/(1 + (2π*τ_0*(f + f_0))^2) + 
            exp(-1im*(dϕ - 2*ϕ_0))*R*τ_1/(1 + (2π*τ_1*(f - f_1))^2) -
            exp(1im*dϕ)*R*τ_1/(1 + (2π*τ_1*(f + f_1))^2)
        )/2
    end

    ## relative to de amplitude ratio R

    function ft_dh_dR_tau_Re(f, A_0, ϕ_0, f_0, τ_0, R, dϕ, f_1, τ_1)
        return A_0*τ_1*(
            exp(-1im*(dϕ - ϕ_0))/(1 + (2π*τ_1*(f - f_1))^2) +
            exp(1im*(dϕ - ϕ_0))/(1 + (2π*τ_1*(f + f_1))^2)
        )/2
    end

    function ft_dh_dR_tau_Im(f, A_0, ϕ_0, f_0, τ_0, R, dϕ, f_1, τ_1)
        return 1im*A_0*τ_1*(
            exp(-1im*(dϕ - ϕ_0))/(1 + (2π*τ_1*(f - f_1))^2) -
            exp(1im*(dϕ - ϕ_0))/(1 + (2π*τ_1*(f + f_1))^2)
        )/2
    end

    ## relative to the phase difference dphi
    function ft_dh_ddphi1_tau_Re(f, A_0, ϕ_0, f_0, τ_0, R, dϕ, f_1, τ_1)
        return -1im*A_0*exp(-1im*(dϕ + ϕ_0))*R*τ_1*(
            exp(2im*ϕ_0)/(1 + (2π*τ_1*(f - f_1))^2)
            - exp(2im*dϕ)/(1 + (2π*τ_1*(f + f_1))^2)
        )/2
    end

    function ft_dh_ddphi1_tau_Im(f, A_0, ϕ_0, f_0, τ_0, R, dϕ, f_1, τ_1)
        return A_0*exp(-1im*(dϕ + ϕ_0))*R*τ_1*(
            exp(2im*ϕ_0)/(1 + (2π*τ_1*(f - f_1))^2)
            + exp(2im*dϕ)/(1 + (2π*τ_1*(f + f_1))^2)
        )/2
    end

    ## relative to the frequency f (with quality factor)
    ### real part
    function ft_dh_dfQ_Re(f, A_lmn, ϕ_lmn, f_lmn, Q_lmn)
        return A_lmn*exp(-1im*ϕ_lmn)*(Q_lmn/π)*((2*f*Q_lmn)^2 - f_lmn^2*(1 + 4*Q_lmn^2))*(
            exp(2im*ϕ_lmn)/(f_lmn^2 + (2*Q_lmn*(f - f_lmn))^2)^2 +
            1/(f_lmn^2 +(2*Q_lmn*(f + f_lmn))^2)^2
        )/2
    end

    function ft_dh_dfQ_Im(f, A_lmn, ϕ_lmn, f_lmn, Q_lmn)
        return 1im*A_lmn*exp(-1im*ϕ_lmn)*(Q_lmn/π)*(f_lmn^2 - 4*Q_lmn^2*(f - f_lmn)*(f + f_lmn))*(
            exp(2im*ϕ_lmn)/(f_lmn^2 + (2*Q_lmn*(f - f_lmn))^2)^2 -
            1/(f_lmn^2 +(2*Q_lmn*(f + f_lmn))^2)^2
        )/2
    end

    ## relative to quality factor Q = pi*f*tau
    ### real part
    function ft_dh_dQ_Re(f, A_lmn, ϕ_lmn, f_lmn, Q_lmn)
        return A_lmn*exp(-1im*ϕ_lmn)*(f_lmn/π)*(
            (f_lmn^2 - (2*Q_lmn*(f + f_lmn))^2)/(f_lmn^2 + (2*Q_lmn*(f + f_lmn))^2)^2 +
            exp(2im*ϕ_lmn)*(f_lmn - 2*(f - f_lmn)*Q_lmn)*(f_lmn + 2*Q_lmn*(f - f_lmn))/(
                f_lmn^2 + (2*Q_lmn*(f - f_lmn))^2
            )^2
        )/2
    end

    function ft_dh_dQ_Im(f, A_lmn, ϕ_lmn, f_lmn, Q_lmn)
        return -1im*A_lmn*exp(-1im*ϕ_lmn)*(f_lmn/π)*(
            (f_lmn^2 - (2*Q_lmn*(f + f_lmn))^2)/(f_lmn^2 + (2*Q_lmn*(f + f_lmn))^2)^2 -
            exp(2im*ϕ_lmn)*(f_lmn - 2*(f - f_lmn)*Q_lmn)*(f_lmn + 2*Q_lmn*(f - f_lmn))/(
                f_lmn^2 + (2*Q_lmn*(f - f_lmn))^2
            )^2
        )/2
    end

    ## relative to amplitude A
    function ft_dh_dAQ_Re(f, A_lmn, ϕ_lmn, f_lmn, Q_lmn)
        return exp(-1im*ϕ_lmn)*f_lmn*(Q_lmn/π)*(
            exp(2im*ϕ_lmn)/(f_lmn^2 + (2*Q_lmn*(f - f_lmn))^2) + 
            1/(f_lmn^2 + (2*Q_lmn*(f + f_lmn))^2)
        )/2
    end

    function ft_dh_dAQ_Im(f, A_lmn, ϕ_lmn, f_lmn, Q_lmn)
        return 1im*exp(-1im*ϕ_lmn)*f_lmn*(Q_lmn/π)*(
            exp(2im*ϕ_lmn)/(f_lmn^2 + (2*Q_lmn*(f - f_lmn))^2) -
            1/(f_lmn^2 + (2*Q_lmn*(f + f_lmn))^2)
        )/2
    end

    ## relative to phase phi
    function ft_dh_dphiQ_Re(f, A_lmn, ϕ_lmn, f_lmn, Q_lmn)
        return 1im*A_lmn*exp(1im*ϕ_lmn)*f_lmn*Q_lmn*(
            exp(2im*ϕ_lmn)/(f_lmn^2 + (2*Q_lmn*(f - f_lmn))^2) -
            1/(f_lmn^2 + (2*Q_lmn*(f + f_lmn))^2)
        )/2
    end

    function ft_dh_dphiQ_Im(f, A_lmn, ϕ_lmn, f_lmn, Q_lmn)
        return - A_lmn*exp(1im*ϕ_lmn)*f_lmn*Q_lmn*(
            exp(2im*ϕ_lmn)/(f_lmn^2 + (2*Q_lmn*(f - f_lmn))^2) +
            1/(f_lmn^2 + (2*Q_lmn*(f + f_lmn))^2)
        )/2
    end
end

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
    using ..FourierTransformsPartial, ..FourierTransformsPartialFH
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

module Quantities
    function trapezio(f, x)
        y = 0
        @fastmath @inbounds @simd for i in 2:length(f)
            y += (x[i] - x[i-1])*(f[i] + f[i-1])/2
        end
        return y
    end

    ## inner product
    function inner_product(f, h1, h2, Sh)
        return 4 * real(trapezio((h1 .* conj(h2)) ./ Sh, f))
    end

    function inner_product_complex(f, h1, h2, Sh)
        return 2 * trapezio((h1 .* conj(h2) + h2 .* conj(h1)) ./ Sh, f)
    end


    function SNR_QNM(freq, noise, ft_Re, ft_Im, F_Re, F_Im)
        ## Compute signal-to-noise-ratio 
        SNR = sqrt(trapezio(4*abs.(F_Re.*ft_Re + F_Im.*ft_Im).^2 ./ noise.^2, freq))
        return SNR
    end

    using QuadGK

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
        #= If Ω_K was not 0
        if Ω_K > 0
            D_M = D_H*sinh(sqrt(Ω_K)*D_C/D_H)/sqrt(Ω_K)
        elseif Ω_K == 0.0
            D_M = D_C
        elseif Ω_K < 0
            D_M = D_H*sin(sqrt(Ω_K)*D_C/D_H)/sqrt(Ω_K)
        end
        D_A = D_M/(1+redshift)
        D_L = (1+redshift)*D_M
        =#
        return D_L
    end

    using DelimitedFiles, Dierckx
    function ImportDetectorStrain(detector, interpolation = false)
        # TODO: Add Einstein Telescope curve 
        # import PSD noise
        ## LISA Strain
        LISA_strain = readdlm("../detectors/LISA_Strain_Sensitivity_range.txt", comments=true, comment_char='#')
        ## LIGO Design sensitivity
        aLIGO_strain = readdlm("../detectors/aLIGODesign.txt", comments=true, comment_char='#')
        ## Cosmic Explorer sensitivity
            ### silica
        CE2silica = readdlm("../detectors/CE/CE2silica.txt", comments=true, comment_char='#')
            ### silicon
        CE2silicon = readdlm("../detectors/CE/CE2silicon.txt", comments=true, comment_char='#')
        ## Einstein Telescope
        ET = readdlm("../detectors/ET/ETDSensitivityCurve.txt", comments=true, comment_char='#')
        
        # choose noise
        noise = Dict()
        if detector == "LIGO"
            noise["freq"], noise["psd"] = aLIGO_strain[:,1], aLIGO_strain[:,2]
            noise["name"] = "Design sensitivity"
            # limit maximum frequency
            noise["psd"] = noise["psd"][(noise["freq"] .< 5000)]
            noise["freq"] = noise["freq"][(noise["freq"] .< 5000)]
    
        elseif detector == "LISA"
            noise["freq"], noise["psd"] = LISA_strain[:,1], LISA_strain[:,2]
            noise["name"] = "LISA sensitivity"
    
        elseif detector == "ET"
            noise["freq"], noise["psd"] = ET[:,1], ET[:,4]
            noise["name"] = "ET_D sum sensitivity"
    
        elseif detector == "CE" || detector == "CE2silicon"
            noise["freq"], noise["psd"] = CE2silicon[:,1], CE2silicon[:,2]
            noise["name"] = "CE silicon sensitivity"
    
        elseif detector == "CE2silica"
            noise["freq"], noise["psd"] = CE2silica[:,1], CE2silica[:,2]
            noise["name"] = "CE silica sensitivity"
        else
            return error("Wrong detector option! Choose \"LIGO\", \"LISA\", \"CE\" = \"CE2silicon\", \"CE2silica\" or \"ET\"")
        end
        if interpolation == false
            return noise
        else
            # interpolate noise curve (Dierckx library)
            itp = Spline1D(noise["freq"], noise["psd"], k = 5, bc = "zero")
            return itp
        end
    end

    export trapezio, inner_product, SNR_QNM, luminosity_distance, ImportDetectorStrain
end