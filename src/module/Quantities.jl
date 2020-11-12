__precompile__()

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
        D_C = D_H*quadgk(x -> 1/E(x), 0, redshift)[1]
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