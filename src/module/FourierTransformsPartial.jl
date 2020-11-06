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