# Add modules to load path
if !("./src/module" in LOAD_PATH)
    push!(LOAD_PATH, "./src/module")
end

using DelimitedFiles, Distributed, HDF5, Random, Distributions, ProgressMeter, Dierckx, PyPlot, FFTW, DSP
using PyCall
using FrequencyTransforms, PhysConstants, Quantities

gwsurrogate = pyimport("gwsurrogate")
np = pyimport("numpy")
import Base.Threads.@spawn



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

function RunAllSXSFolders(M_f, redshift, detector, F_Re, F_Im, q_mass, label, convention = "FH")
	noise = Quantities.ImportDetectorStrain(detector, false)
    folders = readdir("../q_change/")
    
    for simu_folder_name in folders
        if occursin(q_mass, simu_folder_name)
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
            mode_1 = "(2,2,0)"
            mode_2 = "(2,2,1) I"
            modes = ["(2,2,1) I"]#, "(3,3,0)", "(4,4,0)", "(2,1,0)"]

            # Source parameters
            M_final = (1+redshift)*M_f
            M_total = M_final / mass_f
            D_L = Quantities.luminosity_distance(redshift)

            time_unit = (M_total)*PhysConstants.tSun
            strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)

            # Compute surrogate waveform
            sur = gwsurrogate.LoadSurrogate("NRSur7dq4")
            q = parse(Float16, q_mass)
            chi = [0.0, 0.0, 0.0] #spin
            f_ref = 0
            f_low = 0
            D_L = Quantities.luminosity_distance(redshift)
            dt = 0.1
            ellMax = 4
            times = np.arange(0, 100, dt)
            #t_sur, h_sur, dyn_sur = sur(q, chi, chi, dt=dt, f_low=f_low, f_ref=f_ref, ellMax=ellMax, M=M_final, dist_mpc=D_L, units="mks")
            t_sur, h_sur, dyn_sur = sur(q, chi, chi, times = times, f_low = f_low)
            #t_sur .-= 10
            
            t_sur *= time_unit
            dt *= time_unit
            t_sur = vcat(-reverse(t_sur), t_sur)
            for (k,v) in h_sur
                h_sur[k] = strain_unit* v
                h_sur[k] = strain_unit*vcat(reverse(v), v)
            end

            #plot(t_sur, real(h_sur))
            # Fourier transform FH
            ft_Re =  time_unit*strain_unit*abs.(Fourier_1mode.("real", noise["freq"].*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "FH"))
            ft_Im =  time_unit*strain_unit*abs.(Fourier_1mode.("imaginary", noise["freq"].*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "FH"))

            # (2,2,0) + (2,2,1)
            ft_Re_2 =  time_unit*strain_unit*abs.(Fourier_1mode.("real", noise["freq"].*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "FH") +
            Fourier_1mode.("real", noise["freq"].*time_unit, amplitudes[mode_2], phases[mode_2], omega[mode_2][1], omega[mode_2][2], "FH"))
            ft_Im_2 =  time_unit*strain_unit*abs.(Fourier_1mode.("imaginary", noise["freq"].*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "FH") +
            Fourier_1mode.("real", noise["freq"].*time_unit, amplitudes[mode_2], phases[mode_2], omega[mode_2][1], omega[mode_2][2], "FH"))
            #=
        

            # Fourier transform surrogate
            for α in [1, 0.1, 0.01, 0]
            #α = 1   
            # Compute FFT
            ## FFT frequencies
                sur_freqs = FFTW.fftfreq(length(t_sur), 1.0/dt) |> fftshift
                ## tukey window
                sur_tukey_window = DSP.Windows.tukey(length(t_sur), α)
                div = 1
                ## FFT of real part
                sur_fft_re = dt*fft(sur_tukey_window.*real(h_sur[(2,2)]))/div |> fftshift
                ## FFT of imaginary part
                sur_fft_im = dt*fft(sur_tukey_window.*imag(h_sur[(2,2)]))/div |> fftshift
                plot(sur_freqs, 2 .*sur_freqs .*abs.(sur_fft_re), lw = 2, alpha = 0.5, label = "surrogate, α = "*string(α))
            end
            =#

            α = 1   
            # Compute FFT
            ## FFT frequencies
            sur_freqs = FFTW.fftfreq(length(t_sur), 1.0/dt) |> fftshift
            ## tukey window
            sur_tukey_window = DSP.Windows.tukey(length(t_sur), α)
            div = 2
            ## FFT of real part
            sur_fft_re = dt*fft(sur_tukey_window.*real(h_sur[(2,2)]))/div |> fftshift
            ## FFT of imaginary part
            sur_fft_im = dt*fft(sur_tukey_window.*imag(h_sur[(2,2)]))/div |> fftshift

            xscale("log")
            yscale("log")
            #plot(sur_freqs, 2 .*sur_freqs .*abs.(sur_fft_re), lw = 2, color = "green", alpha = 0.5, label = "surrogate, α = 0")
            #plot(sur_freqs, 2 .*sur_freqs .*abs.(sur_fft_re), lw = 3, color = "black", alpha = 0.5, label = "surrogate, α = 0.01")
            plot(sur_freqs, 2 .*sur_freqs .*abs.(sur_fft_im), lw = 2, color = "purple", alpha = 0.5, label = "surrogate, reflect")
            plot(noise["freq"], 2 .*noise["freq"].*ft_Im, label = "FH, "*mode_1)
            plot(noise["freq"], 2 .*noise["freq"].*ft_Im_2, label = "FH, (2,2,0) + (2,2,1)")
            
            xlim(1e0, 1e3)
            #ylim(1e-27, 1e-22)
            xlabel("freq [Hz]")
            ylabel("Characteristic strain")
            title("t_0 = pico, Massa = "*string(M_f)*"M⊙, redshift = "*string(redshift))
            legend()

            #close("all")
            #plot(t_sur, h_sur[(2,2)])

        end
    end
end


